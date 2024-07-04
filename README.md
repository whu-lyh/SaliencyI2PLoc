## SaliencyI2PLoc

### Installation

```bash
cd scripts
sh install.sh
```
> You may required to change the coding manner of sh files using `sed -i "s/\r//" *.sh` to avoid the file unrecognition.

+ both pytorch1.13.1-cuda11.6 and pytorch2.1.2-cuda12.1 works


### Train

```bash
cd scripts
sh train.sh # single machine
#sh train_dist.sh # DDP mode
```

The configuration information will be loaded all in once from the CrossModalityRetrieval.yaml file, including the optimizer, scheduler, dataset, model and other information.

### Test

```bash
cd scripts
sh test.sh
```
