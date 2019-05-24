from torch.utils.data import Dataset, DataLoader

from fastai.vision.data import ImageItemList
from fastai.vision.image import Image, pil2tensor
from fastai.data_block import get_files
from fastai.basics import *
from fastai.data_block import PreProcessors
from fastai.vision.data import ImageDataBunch, channel_view, normalize_funcs
from fastai.vision import Image
from fastai.basic_data import DeviceDataLoader
from fastai.vision import open_image

import numpy as np

from PIL import Image as PILImage

def loss_func(ys, margin=1.00, l2=0.01):
    z_p, z_n, z_d = ys

    l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
    l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
    l_nd = l_n + l_d
    loss = F.relu(l_n + l_d + margin)
    l_n = torch.mean(l_n)
    l_d = torch.mean(l_d)
    l_nd = torch.mean(l_n + l_d)
    loss = torch.mean(loss)
    if l2 != 0:
        loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
    return loss


class MultiImageDeviceDataLoader(DeviceDataLoader):
    def proc_batch(self,b:Tensor)->Tensor:
        "Process batch `b` of `TensorImage`."
        b = to_device(b, self.device)
        
        xs, y = b
        for f in listify(self.tfms):
            # self.tfms contains a set of transforms for each x in xs
            if isinstance(f, list):
                for (f_, x) in zip(f, xs):
                    x = f_((x, y))
            else:
                for x in xs:
                    x = f((x, y))
        return b

def normalize(x:TensorImage, mean:FloatTensor,std:FloatTensor)->TensorImage:
    "Normalize `x` with `mean` and `std`."
    return (x-mean[...,None,None]) / std[...,None,None]

def denormalize(x:TensorImage, mean:FloatTensor,std:FloatTensor, do_x:bool=True)->TensorImage:
    "Denormalize `x` with `mean` and `std`."
    return x.cpu().float()*std[...,None,None] + mean[...,None,None] if do_x else x.cpu()

def _normalize_batch(b:Tuple[Tensor,Tensor], mean:FloatTensor, std:FloatTensor, do_x:bool=True, do_y:bool=False)->Tuple[Tensor,Tensor]:
    "`b` = `x`,`y` - normalize `x` array of imgs and `do_y` optionally `y`."
    x,y = b
    mean,std = mean.to(x.device),std.to(x.device)
    if do_x: x = normalize(x,mean,std)
    if do_y and len(y.shape) == 4: y = normalize(y,mean,std)
    return x,y

def normalize_funcs(mean:FloatTensor, std:FloatTensor, do_x:bool=True, do_y:bool=False)->Tuple[Callable,Callable]:
    "Create normalize/denormalize func using `mean` and `std`, can specify `do_y` and `device`."
    mean,std = tensor(mean),tensor(std)
    return (partial(_normalize_batch, mean=mean, std=std, do_x=do_x, do_y=do_y),
            partial(denormalize, mean=mean, std=std, do_x=do_x))

    
class MultiImageDataBunch(ImageDataBunch):
    _ddl_cls = MultiImageDeviceDataLoader
    
    def __init__(self, train_dl:DataLoader, valid_dl:DataLoader, fix_dl:DataLoader=None, test_dl:Optional[DataLoader]=None,
                 device:torch.device=None, dl_tfms:Optional[Collection[Callable]]=None, path:PathOrStr='.',
                 collate_fn:Callable=data_collate, no_check:bool=False):
        self.dl_tfms = listify(dl_tfms)
        self.device = defaults.device if device is None else device
        assert not isinstance(train_dl,self._ddl_cls)
        def _create_dl(dl, **kwargs):
            if dl is None: return None
            return self._ddl_cls(dl, self.device, self.dl_tfms, collate_fn, **kwargs)
        self.train_dl,self.valid_dl,self.fix_dl,self.test_dl = map(_create_dl, [train_dl,valid_dl,fix_dl,test_dl])
        if fix_dl is None: self.fix_dl = self.train_dl.new(shuffle=False, drop_last=False)
        self.single_dl = _create_dl(DataLoader(valid_dl.dataset, batch_size=1, num_workers=0))
        self.path = Path(path)
        if not no_check: self.sanity_check()
    
    def batch_stats(self, funcs:Collection[Callable]=None)->Tensor:
        "Grab a batch of data and call reduction function `func` per channel"
        funcs = ifnone(funcs, [torch.mean,torch.std])
        #x = self.one_batch(ds_type=DatasetType.Valid, denorm=False)[0].cpu()
        
        # one_batch gives (x,y) pair on first dim, next dim is going to be the number of images
        # xs = [b.cpu() for b in self.one_batch(ds_type=DatasetType.Valid, denorm=False)[0]]
        # return [[func(channel_view(x), 1) for func in funcs] for x in xs]
        
        x = self.one_batch(ds_type=DatasetType.Valid, denorm=False)[0][0].cpu()
        return [func(channel_view(x), 1) for func in funcs]

    def show_batch(self, rows:int=5, ds_type:DatasetType=DatasetType.Train, reverse:bool=False, **kwargs)->None:
        raise NotImplementedError("Leif: haven't made this work with the triplet trainer yet")


class NPMultiImageItemList(ImageItemList): 
    c = 100
    
    _bunch = MultiImageDataBunch
    class ImagesList(list):
        def __init__(self, fn, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fn = fn
            self.id = fn.split('/')[-1].split('_')[0]

        @property
        def size(self):
            return self[0].size
        
        def apply_tfms(self, tfms, **kwargs):
            items = []
            for item in self:
                items.append(item.apply_tfms(tfms, **kwargs))
                
            return items

    def open(self, fn, div:bool=True):
        
        fn = str(fn)
        
        images = self.ImagesList(fn)

        
        fns = [
            fn,
            fn.replace('anchor', 'neighbor'),
            fn.replace('anchor', 'distant')
        ]
        
        for fn_ in fns:
            if fn.endswith('.npy'):
                x = np.load(fn_)[:,:,:-1]  # NAIP -> RGB (4 -> 3 channels)
                x = PILImage.fromarray(x).convert('RGB')  # need to make the image RGB here otherwise we get an alpha channel
            
                # copied from fastai.vision.image.open_image
                x = pil2tensor(x,np.float32)
                if div: x.div_(255)
                    
                images.append(Image(x))
            else:
                x = open_image(fn_)
                images.append(x)

            
            
        return images
    

    
    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=None,
                    recurse:bool=True, include:Optional[Collection[str]]=None,
                    processor:PreProcessors=None, **kwargs)->ItemList:
        path = Path(path)
        
        files = get_files(path, extensions, recurse=recurse, include=include)
        
        # only get our anchor files for now
        files = filter(lambda p: 'anchor' in p.name, files)
               
       # return cls(items=files, path=path, processor=processor, **kwargs)
        return cls(files, path=path, processor=processor, **kwargs)

def loss_batch(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None, opt:OptOptimizer=None,
               cb_handler:Optional[CallbackHandler]=None)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]
    out = [model(x) for x in xb]
    out = cb_handler.on_loss_begin(out)
        
    #out = cb_handler.on_loss_begin(out)

    if not loss_func: return to_detach(out), yb[0].detach()
    
    #print(out)
    loss = loss_func(out)

    if opt is not None:
        loss = cb_handler.on_backward_begin(loss)
        loss.backward()
        cb_handler.on_backward_end()
        opt.step()
        cb_handler.on_step_end()
        opt.zero_grad()

    return loss.detach().cpu()


def monkey_patch_fastai():
    try:
        print(loss_batch_orig)
    except:
        import fastai.basic_train
        loss_batch_orig = fastai.basic_train.loss_batch
    fastai.basic_train.loss_batch = loss_batch
