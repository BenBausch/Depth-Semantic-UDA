import unittest
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataloaders import get_dataset
import  dataloaders.kitti.dataset_kitti as kitti
from cfg.config_single_dataset import get_cfg_defaults  # local variable usage pattern, or:

class Test_PathProcessing(unittest.TestCase):
    def setUp(self):
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file("data/kitti_all_dataset_test.yaml")
        self.cfg.freeze()

        self.path_rgb_dummy = '/home/petek/kalimu_depricated/data/kitti_depricated/base/train/2011_09_26_drive_0001_sync/rgb/image_02/data/0000000106.png'

    def test_decompose_rgb_path(self):
        mode, drive_seq, cam_id, frame_id, format = kitti.decompose_rgb_path(self.path_rgb_dummy)
        assert mode == 'train'
        assert drive_seq == '2011_09_26_drive_0001_sync'
        assert cam_id == '2'
        assert frame_id == '0000000106'
        assert format == '.png'

class Test_PathsKitti(unittest.TestCase):
    def setUp(self):
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file("data/kitti_all_dataset_test.yaml")
        self.cfg.freeze()

        self.mode = 'train'
        self.pathsObj = kitti._PathsKittiFromMotion(self.mode, self.cfg.dataset.split, self.cfg)

        self.nof_rgb =  56278
        self.nof_sparse = 53938
        self.nof_gt = 53938

    def test_nof_files(self):
        rgb_paths = self.pathsObj.get_rgb_image_paths(self.mode)
        gt_paths = self.pathsObj.get_gt_depth_image_paths(self.mode)
        sparse_paths = self.pathsObj.get_sparse_depth_image_paths(self.mode)

        assert len(rgb_paths) == self.nof_rgb
        assert len(gt_paths) == self.nof_gt
        assert len(sparse_paths) == self.nof_sparse


    def test_make_consistent(self):
        # This test is somehow redundant, as there are already some checks in the constructor...
        # The constructor already checks whether the datasets are consistent. If this point in the code is reached, it means
        # that they are consistent, otherwise an assertion would've been thrown
        assert self.pathsObj.is_consistent() == True


    def test_is_consistent_gt_missing(self):
        # Make the datasets inconsistent by deleting a sparse image
        self.pathsObj.paths_depth_gt.pop(6)
        assert self.pathsObj.is_consistent() == False


    def test_is_consistent_rgb_missing(self):
        # Make the datasets inconsistent by deleting an rgb image
        self.pathsObj.paths_rgb.pop(5)
        assert self.pathsObj.is_consistent() == False


    def test_is_consistent_sparse_missing(self):
        # Make the datasets inconsistent by deleting a sparse image
        self.pathsObj.paths_depth_sparse.pop(12)
        assert self.pathsObj.is_consistent() == False

# The following part is just to visualize the data and to check visually whether thins look correct. Comment it out
# if you only want to test the functionalities (see code above)
class Test_DatasetKitti(unittest.TestCase):
    def setUp(self):
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file("data/kitti_all_dataset_test.yaml")
        self.cfg.freeze()

        train_dataset = get_dataset(self.cfg.dataset.name, "train", self.cfg.dataset.split, self.cfg)
        val_dataset = get_dataset(self.cfg.dataset.name, "val", self.cfg.dataset.split, self.cfg)

        self.train_loader = DataLoader(train_dataset, batch_size=self.cfg.train.batch_size, shuffle=True, num_workers=self.cfg.train.nof_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.cfg.val.batch_size, shuffle=True, num_workers=self.cfg.val.nof_workers, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        for i, batch_data in enumerate(self.train_loader):
            print(i, "-th run")
            rgb_curr = batch_data[('rgb', self.cfg.train.rgb_frame_offsets[0])]
            rgb_prev = batch_data[('rgb', self.cfg.train.rgb_frame_offsets[1])]
            rgb_next = batch_data[('rgb', self.cfg.train.rgb_frame_offsets[2])]
            gt = batch_data["gt"]
            sparse = batch_data["sparse"]

            plt.imshow(rgb_curr.squeeze().permute(1, 2, 0))
            plt.title("Current rgb image ({} offset)".format(self.cfg.train.rgb_frame_offsets[0]))
            plt.show()

            plt.imshow(rgb_prev.squeeze().permute(1, 2, 0))
            plt.title("Previous rgb image ({} offset)".format(self.cfg.train.rgb_frame_offsets[1]))
            plt.show()

            plt.imshow(rgb_next.squeeze().permute(1, 2, 0))
            plt.title("Next rgb image (+{} offset)".format(self.cfg.train.rgb_frame_offsets[2]))
            plt.show()

            plt.imshow(sparse.squeeze(dim=0).permute(1, 2, 0))
            plt.title("Sparse depth image (0 offset)")
            plt.show()

            plt.imshow(gt.squeeze(dim=0).permute(1, 2, 0))
            plt.title("GT depth image (0 offset)")
            plt.show()

        for i, batch_data in enumerate(self.val_loader):

            print(i, "-th run")
            rgb_curr = batch_data[('rgb', opts.rgb_frame_offsets[0])]
            plt.imshow(rgb_curr.squeeze().permute(1, 2, 0))
            plt.title("Current rgb image ({} offset)".format(self.cfg.train.rgb_frame_offsets[0]))
            plt.show()

            try:
                rgb_prev = batch_data[('rgb', opts.rgb_frame_offsets[1])]
                plt.imshow(rgb_prev.squeeze().permute(1, 2, 0))
                plt.title("Previous rgb image ({} offset)".format(self.cfg.train.rgb_frame_offsets[1]))
                plt.show()
            except:
                print("Image (prev) seems to not exist (i.e. first image of the driving sequence is being processed!)")

            try:
                rgb_next = batch_data[('rgb', opts.rgb_frame_offsets[2])]
                plt.imshow(rgb_next.squeeze().permute(1, 2, 0))
                plt.title("Next rgb image (+{} offset)".format(self.cfg.train.rgb_frame_offsets[2]))
                plt.show()
            except:
                print("Image (next) seems to not exist (i.e. last image of the driving sequence is being processed!)")

            gt = batch_data["gt"]
            sparse = batch_data["sparse"]

            plt.imshow(sparse.squeeze(dim=0).permute(1, 2, 0))
            plt.title("Sparse depth image (0 offset)")
            plt.show()

            plt.imshow(gt.squeeze(dim=0).permute(1, 2, 0))
            plt.title("GT depth image (0 offset)")
            plt.show()

if "__main__" == __name__:
    unittest.main()