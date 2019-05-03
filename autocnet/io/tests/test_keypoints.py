import numpy as np
import pandas as pd
import pytest

from plio.io import io_hdf

from .. import keypoints

@pytest.fixture
def kd(scope='module'):
    kps = pd.DataFrame(np.random.random((128,3)), columns=['a', 'b', 'c'])
    desc = np.random.random((128,128))

    return kps, desc

def test_read_write_npy(tmpdir, kd):
    kps, desc = kd
    path = tmpdir.join('out.npz')
    keypoints.to_npy(kps, desc, path.strpath)
    reloaded_kps, reloaded_desc = keypoints.from_npy(path.strpath)

    assert reloaded_kps.equals(kps)
    np.testing.assert_array_equal(reloaded_desc, desc)

def test_read_write_hdf(tmpdir, kd):
    kps, desc = kd
    path = tmpdir.join('out.h5')
    keypoints.to_hdf(path.strpath, keypoints=kps, descriptors=desc)
    reloaded_kps, reloaded_desc = keypoints.from_hdf(path.strpath)

    assert reloaded_kps.equals(kps)
    np.testing.assert_array_equal(reloaded_desc, desc)

    reloaded_kps = keypoints.from_hdf(path.strpath, descriptors=False)
    assert reloaded_kps.equals(kps)

    reloaded_desc = keypoints.from_hdf(path.strpath, keypoints=False)
    np.testing.assert_array_equal(reloaded_desc, desc)

def test_read_hdf_with_index(tmpdir, kd):
    kps, desc = kd                                                                 
    path = tmpdir.join('out.h5')                                                   
    keypoints.to_hdf(path.strpath, keypoints=kps, descriptors=desc) 

    reloaded_kps, reloaded_desc = keypoints.from_hdf(path.strpath, index=np.arange(10))
    assert len(reloaded_kps) == len(reloaded_desc) == 10
    np.testing.assert_array_equal(np.arange(10), reloaded_kps.index.values)

def test_read_write_hdf_with_live_file(tmpdir, kd):
    kps, desc = kd
    path = tmpdir.join('live.h5')
    hf = io_hdf.HDFDataset(path.strpath, mode='w')
    keypoints.to_hdf(hf, keypoints=kps, descriptors=desc)
    reloaded_kps, reloaded_desc = keypoints.from_hdf(hf)

    assert reloaded_kps.equals(kps)
    np.testing.assert_array_equal(reloaded_desc, desc)

@pytest.mark.parametrize("filename, outdir, expected",
                         [('foo', None, 'foo_kps.h5'),
                          ('/path/foo', None, '/path/foo_kps.h5'),
                          ('foo', '/path', '/path/foo_kps.h5')
                         ])
def test_create_output_path(filename, outdir, expected):
    assert keypoints.create_output_path(filename, outdir=outdir) == expected
