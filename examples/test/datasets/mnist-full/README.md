# Complete MNIST dataset

The four original IDX gzip files are mirrored by CVDF at
https://github.com/cvdfoundation/mnist. Their published MD5 hashes are checked
in `tests/test_example_datasets.py`. The first 55,000 official training images
are the training split, the last 5,000 are validation, and all 10,000 official
test images form the test split. Files are bundled for offline training.
