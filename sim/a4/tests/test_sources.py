import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from sim.a4.sources import safe_extract

class Archives(unittest.TestCase):
    def test_reject_escaping_and_links(self):
        for name,kind in [('../escape',tarfile.REGTYPE),('/absolute',tarfile.REGTYPE),('link',tarfile.SYMTYPE),('C:/drive',tarfile.REGTYPE)]:
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); archive=root/'fixture.tar'
                with tarfile.open(archive,'w') as stream:
                    member=tarfile.TarInfo(name); member.type=kind; member.linkname='../escape'; member.size=0
                    stream.addfile(member,io.BytesIO())
                with self.assertRaises(ValueError): safe_extract(archive,root/'out')

    def test_upstream_shaped_tiny_fixture(self):
        import torch
        from sim.a4.data import load_prefix
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'prefix_sums_data'; root.mkdir()
            x=torch.tensor([[0.,1.],[1.,1.]])
            torch.save(x,root/'2_data.pth'); torch.save(x.cumsum(1).remainder(2),root/'2_targets.pth')
            data=load_prefix(root,2,role='train',fixture=True)
            self.assertEqual(data['x'].shape,(2,1,2))
