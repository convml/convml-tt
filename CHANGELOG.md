# Changelog

## [v0.7.1](https://github.com/convml/convml_tt/tree/HEAD)

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
