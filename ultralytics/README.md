# Changes to introduce multispectral channels in ultralytics

- `/cfg/default.yaml`

```Python
129. bands_to_apply:
```

- `/utils/__init__.py`

```Python
1081.  from .utils import is_vegetation_index, get_band_combination
```


- `/data/base.py`
```Python
19.  from ultralytics.utils import is_vegetation_index, get_band_combination

64.  bands_to_apply=None,

82.  self.bands_to_apply = bands_to_apply

160. path = self.im_files[i]
161. if self.bands_to_apply and self.bands_to_apply != ["RGB"]:
162.    bands = []
163.    root_dir = os.path.dirname(path)
164.    imgName = Path(path).stem
165.    for band_name in self.bands_to_apply:
166.        if band_name == 'RGB' or band_name == 'RGB'.lower():
167.            im_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
168.            bands.append(im_rgb)
169.        elif is_vegetation_index(band_name):
170.            ms_image = []
171.            for band in ['Red', 'Green', 'Blue', 'RE', 'NIR']:
172.                ms_image.append(cv2.imread(os.path.join(root_dir, f'{imgName}_{band}.TIF'),cv2.IMREAD_GRAYSCALE))
173.            ms_image = np.dstack(ms_image)
174.            im_vi = get_band_combination(ms_image,band_name)
175.            #im_vi = (im_vi / 255).astype(np.uint8)
176.            #cv2.imwrite('RESULT.JPG',im_vi)
177.            bands.append(im_vi)
178.        else:
179.            bands.append(cv2.imread(os.path.join(root_dir, f'{imgName}_{band_name}.TIF'), cv2.IMREAD_GRAYSCALE))
180.    im = np.dstack(bands)
181. else:
182.    im = cv2.imread(path)  # BGR
183. assert im is not None, 'Image Not Found ' + path
```

- `/data/build.py`
```Python
84.  def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False, bands_to_apply=None):

103.    bands_to_apply=bands_to_apply
```

- `/data/utils.py`
```Python
39. #IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
40. IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "webp", "pfm"}  # image suffixes
```

- `/engine/trainer.py`
```Python
107. # Multispectral
152. self.bands_to_apply = self.args.bands_to_apply
285. self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode="train", bands_to_apply=self.bands_to_apply)
289.  self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val", bands_to_apply=self.bands_to_apply
```

- `/engine/validator.py`
```Python
105.  # Multispectral
106.  self.bands_to_apply = self.args.bands_to_apply
157. self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch, bands_to_apply=self.bands_to_apply)
```

- `/models/yolo/detect/train.py`: Get the number of channels of the model when init from the data.yaml (`self.data["ch"]`) file.
```Python
33. def build_dataset(self, img_path, mode="train", batch=None, bands_to_apply=None):
43. return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, bands_to_apply=bands_to_apply)
45. def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train", bands_to_apply=None):
49. dataset = self.build_dataset(dataset_path, mode, batch_size, bands_to_apply=bands_to_apply)
88. model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["ch"], verbose=verbose and RANK == -1)
```

- `/models/yolo/detect/val.py`
```Python
216. def build_dataset(self, img_path, mode="val", batch=None, bands_to_apply=None):
225. return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride, bands_to_apply=bands_to_apply)
227. def get_dataloader(self, dataset_path, batch_size, bands_to_apply=None):
229. dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val", bands_to_apply=bands_to_apply)
```

- `/nn/tasks.py`: Get the number of channels of the model when init from the data.yaml file.
```Python
283. #ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
```

- `/utils/utils.py`: Add this file based on https://github.com/ManfredGonzalez/multispectral-pineapple-detection/blob/main/yolov5/utils/utils.py

## Examples

> Example on how to run for multispectral images (add to the parameter `bands_to_apply`, the list of channels strings, and in `data.yaml` update the parameter `ch:NUMBER_CHANNELS`): `../train_yolov9_multispectral.py`

> Example on how to run for RGB images (add to the parameter `bands_to_apply`, `RGB` and in `data.yaml` update the parameter `ch:3`): `../train_yolov9.py`