# Changelog

## [v0.11.0](https://github.com/convml/convml_tt/tree/HEAD)

[Full Changelog](https://github.com/convml/convml_tt/compare/v0.10.1...v0.11.0)

*new features*

- Annotated scatterplots now work for `data.dataset.MovingWindowImageTilingDataset`s
  [\#52](https://github.com/convml/convml_tt/pull/52)

- `utils.get_embeddings` now uses GPU for producing the embeddings if one is
  available, drastically speeding up inference time.
  [\#53](https://github.com/convml/convml_tt/pull/53)

*breaking changes*

- `utils.make_sliding_tile_model_predictions` has been removed in favour of
  using `utils.get_embeddings` directly
  [\#54](https://github.com/convml/convml_tt/pull/54)


## [v0.10.1](https://github.com/convml/convml_tt/tree/v0.10.1)

[Full Changelog](https://github.com/convml/convml_tt/compare/v0.10.0...v0.10.1)

*bugfixes*

- Fix for xarray bug which arised with call in
  `utils.interpretation.rectpred.plot.make_rgb`
  [\#51](https://github.com/convml/convml_tt/pull/51)


## [v0.10.0](https://github.com/convml/convml_tt/tree/v0.10.0)

[Full Changelog](https://github.com/convml/convml_tt/compare/v0.9.0...v0.10.0)

*new features*

- Add writer for tensorboard projector visualizer to enable exploration of
  tiles in embedding space
  [\#43](https://github.com/convml/convml_tt/pull/43)

- support for z-order and zero-offset annotations in annotated scatterplot
  [\#44](https://github.com/convml/convml_tt/pull/44)

- add isomap to available transforms on rectangular domain
  [\#47](https://github.com/convml/convml_tt/pull/47)

- make it possible to load fastai v1 model using
  `TripletTrainerModel.load_from_checkpoint`
  [\#46](https://github.com/convml/convml_tt/pull/46)

- add `data.dataset.MovingWindowImageTilingDataset` to produce image tile
  data from a sliding window across a larger image
  [\#45](https://github.com/convml/convml_tt/pull/45)

*breaking changes*

- `utils.make_sliding_tile_model_predictions` now takes in
  a `data.dataset.MovingWindowImageTilingDataset` rather than an image,
  and so a dataset must be created from an image first
  [\#45](https://github.com/convml/convml_tt/pull/45)

*changed default*

- annotated scatterplot now sets equal x/y-axis
  [\#44](https://github.com/convml/convml_tt/pull/44)

*maintenance*

- fix for `pytorch-ligtning >= 1.5.0` and exclude `torchvision == 0.10.*`
  (can't load `.tgz` files)
  [\#48](https://github.com/convml/convml_tt/pull/48)

- Add link to Google Colab instructions
  [\#48](https://github.com/convml/convml_tt/pull/49)



## [v0.9.0](https://github.com/convml/convml_tt/tree/v0.9.0)

[Full Changelog](https://github.com/convml/convml_tt/compare/v0.8.0...v0.9.0)

*new features*

- Add before and after training logger for producing a dendrogram plot
  [\#40](https://github.com/convml/convml_tt/pull/40)

- Add "best triplet" option to tile sampling method for dendrogram plot
  which uses triplets for which anchor-neighbor are the closest
  [\#38](https://github.com/convml/convml_tt/pull/38)

- Add rectpred sample plot function [\#30](https://github.com/convml/convml_tt/pull/30)

- Enable dendrogram plots for triplets [\#35](https://github.com/convml/convml_tt/pull/35)

- Add interactive rect plot for visualising embedding distance to
  a selected part of the domain [\#34](https://github.com/convml/convml_tt/pull/34)

- Add one-cycle learning rate scheduler
  [\#32](https://github.com/convml/convml_tt/pull/32)

*changed default*

- Weight decay (L2-regularisation) is now enabled by default (set to
  `0.01`) and the learning-rate has been increased to `1.0e-2` to match
  the values used when training the fastai v1 based model. Note this
  learning-rate was optimised for for a batch-size of `50` using the
  `resnet18` architecture. For different batch-size, weight-decay or
  architecture a new learning may be needed (use the [pytorch lightning
  learning rate
  finder](https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html))
  [\#30](https://github.com/convml/convml_tt/pull/30)

*bugfixes*

- Fix for grid-overview plot always showing same set of tiles rather than tiles with indecies provided [\#36](https://github.com/convml/convml_tt/pull/36)

## [v0.8.0](https://github.com/convml/convml_tt/tree/v0.8.0)

[Full Changelog](https://github.com/convml/convml_tt/compare/v0.7.1...v0.8.0)

**Merged pull requests:**

- Add support for anti-aliased backbones [\#28](https://github.com/convml/convml_tt/pull/28) ([leifdenby](https://github.com/leifdenby))
- Rectpred refactor [\#27](https://github.com/convml/convml_tt/pull/27) ([leifdenby](https://github.com/leifdenby))
- Bugfix for load of fastai v1 trained model [\#26](https://github.com/convml/convml_tt/pull/26) ([leifdenby](https://github.com/leifdenby))

## [v0.7.1](https://github.com/convml/convml_tt/tree/v0.7.1)

Bugfixes and minor improvements

[Full Changelog](https://github.com/convml/convml_tt/compare/v0.7.0...v0.7.1)

**Merged pull requests:**

- Add black code-check to CI [\#25](https://github.com/convml/convml_tt/pull/25) ([leifdenby](https://github.com/leifdenby))
- CLI bugfixes [\#24](https://github.com/convml/convml_tt/pull/24) ([leifdenby](https://github.com/leifdenby))
- improvements to CLI [\#23](https://github.com/convml/convml_tt/pull/23) ([leifdenby](https://github.com/leifdenby))

## [v0.7.0](https://github.com/convml/convml_tt/tree/v0.7.0)

Model architecture and training rewritten to use [pytorch-lightning](https://pytorchlightning.ai/).

Specifically this includes:

- `pytorch.Dataset` (`convml_tt.data.TripletDataset`) for handling loading of individual triplet datasets (and one for single-tile datasets). This has the option to load all data into a single numpy memory-mapped array to reduce number of disc reads

- `pytorch_lightning.LightningDataModule` (`convml_tt.data.TripletDataModule`) for handling transform, batching and splitting data for train/test

  - includes transforms previously provided by fastai now using the
    [kornia](https://kornia.github.io/) library
  - correct normalisation when using pretrained network is used

- `pytorch_lightning.LightningModule` (`convml_tt.system.TripletTrainerModel`) which handles the training logic: building model, setting learning rate, batch size. This is now flexible in the number of embedding dimensions used

- Command-line interface for doing training (`convml_tt.trainer`)

- functionality for downloading and unpacking example datasets

- refactoring of utility functions and inference plots to ensure they work with
  the new architecture

- ARC3 HPC and JASMIN HPC systems submission scripts (SGE and SLURM) for running training on single GPU

[Full Changelog](https://github.com/convml/convml_tt/compare/v0.6.0...v0.7.0)

**Implemented enhancements:**

- Simplifying software dependencies [\#14](https://github.com/convml/convml_tt/issues/14)
- Fixes and notes for training on JASMIN [\#20](https://github.com/convml/convml_tt/pull/20) ([leifdenby](https://github.com/leifdenby))
- Switch from fastai to pytorch-lightning [\#17](https://github.com/convml/convml_tt/pull/17) ([leifdenby](https://github.com/leifdenby))

## [v0.6.0](https://github.com/convml/convml_tt/tree/v0.6.0) (2021-02-24)

Last version using fastai v1. Contains model as used in L Denby 2020.

[Full Changelog](https://github.com/convml/convml_tt/compare/v0.5.0...v0.6.0)

**Merged pull requests:**

- Setup CI for windows, macos and linux [\#18](https://github.com/convml/convml_tt/pull/18) ([leifdenby](https://github.com/leifdenby))
- Simplifying installation of convml\_tt [\#15](https://github.com/convml/convml_tt/pull/15) ([leifdenby](https://github.com/leifdenby))
- Tile generation and analysis pipeline [\#12](https://github.com/convml/convml_tt/pull/12) ([leifdenby](https://github.com/leifdenby))
- add test for creating embedding from tile list [\#11](https://github.com/convml/convml_tt/pull/11) ([leifdenby](https://github.com/leifdenby))
