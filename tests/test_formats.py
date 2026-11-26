from __future__ import annotations
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageCms, JpegImagePlugin

from enhancer.core import load_texture, Settings
from enhancer.errors import EnhancerError
from enhancer.formats import decode_source, encode_original, preflight, verify_export
from enhancer.formats import dds,tga,png
from enhancer.storage import save_original_atomic
from format_fixtures import pattern,dds_fixture,raw_dds,tga_fixture,png_fixture,png_chunk


class FormatCase(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='formats spaces ü ')
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def texture(self,data,suffix):
        path=self.root/('source'+suffix);path.write_bytes(data);return load_texture(path)
    def roundtrip(self,data,suffix):
        texture=self.texture(data,suffix)
        target=texture.image.resize(tuple(v*2 for v in texture.image.size),Image.Resampling.NEAREST)
        encoded=encode_original(target,texture.profile)
        verify_export(encoded,texture.profile,target.size)
        return texture,target,encoded


class DdsFormats(FormatCase):
    def test_dxt_compression_and_all_mip_policies(self):
        for codec in ('DXT1','DXT3','DXT5','DXT2','DXT4'):
            for levels,expected in ((1,1),(3,3),(5,6)):
                with self.subTest(codec=codec,levels=levels):
                    texture,target,output=self.roundtrip(dds_fixture(codec=codec,levels=levels,premul=codec in('DXT2','DXT4')),'.DDS')
                    info=dds.inspect(output)
                    self.assertEqual(info.mipmaps,expected)
                    self.assertEqual(output[84:88],codec.encode())
                    self.assertEqual(output[76:108],texture.profile.details.header[76:108])
                    self.assertEqual(struct.unpack_from('<I',output,20)[0],info.level_size(*target.size))
                    self.assertEqual(len(output),len(info.header)+sum(info.level_size(max(1,target.width>>i),max(1,target.height>>i)) for i in range(expected)))
    def test_uncompressed_masks_and_pitch(self):
        variants=[(32,(0xff0000,0xff00,0xff,0xff000000),'rgb'),(32,(0xff,0xff00,0xff0000,0xff000000),'rgb'),
            (32,(0xff0000,0xff00,0xff,0),'rgb'),(24,(0xff0000,0xff00,0xff,0),'rgb'),
            (16,(0xf800,0x7e0,0x1f,0),'rgb'),(16,(0x7c00,0x3e0,0x1f,0x8000),'rgb'),
            (16,(0xf00,0xf0,0xf,0xf000),'rgb'),(8,(0xe0,0x1c,3,0),'rgb'),
            (8,(255,0,0,0),'luminance'),(16,(255,0,0,65280),'luminance'),(8,(0,0,0,255),'alpha')]
        for bits,masks,kind in variants:
            for align in (1,4):
                with self.subTest(bits=bits,masks=masks,kind=kind,align=align):
                    texture,target,output=self.roundtrip(raw_dds(bits=bits,masks=masks,kind=kind,alignment=align,levels=3),'.dds')
                    info=dds.inspect(output)
                    decoded=dds.decode(output,info)
                    self.assertEqual(decoded.size,target.size)
                    # exact nearest-neighbor roundtrip for channels already quantized at load
                    self.assertEqual(decoded.tobytes(),target.tobytes())
    def test_dx10_unorm_srgb_and_premultiplied(self):
        for code,codec in ((70,'DXT1'),(71,'DXT1'),(72,'DXT1'),(73,'DXT3'),(74,'DXT3'),(75,'DXT3'),(76,'DXT5'),(77,'DXT5'),(78,'DXT5')):
            for premul in (False,True):
                with self.subTest(code=code,premul=premul):
                    tex,target,out=self.roundtrip(dds_fixture(codec=codec,dxgi=code,premul=premul,levels=3),'.dds')
                    self.assertEqual(out[128:148],tex.profile.details.header[128:148])
                    self.assertEqual(dds.inspect(out).premultiplied,premul)
    def test_bc4_bc5_channels_and_fourcc_retained(self):
        for fourcc,channels in ((b'ATI1',1),(b'BC4U',1),(b'ATI2',2),(b'BC5U',2)):
            raw=dds_fixture('DXT5',size=(8,8))
            block=np.frombuffer(raw[128:],dtype='uint8').reshape(-1,16)[:,:8]
            body=block.tobytes() if channels==1 else np.concatenate((block,block),axis=1).tobytes()
            header=bytearray(raw[:128]);header[84:88]=fourcc
            tex,target,out=self.roundtrip(bytes(header)+body,'.dds')
            self.assertEqual(out[84:88],fourcc)
            self.assertTrue(tex.profile.details.data_texture)
            with self.assertRaises(EnhancerError):preflight(tex.profile,tex.image,method='openai')
            preflight(tex.profile,tex.image,method='nearest')
    def test_dx10_raw_layouts(self):
        for code,(bits,masks,kind) in dds.DXGI_RAW.items():
            if kind=='rg':continue
            base=raw_dds(bits=bits,masks=masks,kind=kind)
            head=bytearray(base[:128]);struct.pack_into('<I',head,80,4);head[84:88]=b'DX10'
            data=bytes(head)+struct.pack('<5I',code,3,0,1,1)+base[128:]
            with self.subTest(code=code):
                tex,target,out=self.roundtrip(data,'.dds')
                self.assertEqual(out[128:148],data[128:148])
    def test_dimensions_and_missing_mip_payload_rejected(self):
        valid=dds_fixture(levels=5)
        for mutated in (valid[:127],valid[:-1],valid+b'extra',b'DDS '):
            with self.subTest(length=len(mutated)),self.assertRaises(EnhancerError):self.texture(mutated,'.dds')
    def test_invalid_header_sizes(self):
        for offset in(4,76):
            data=bytearray(dds_fixture());struct.pack_into('<I',data,offset,99)
            with self.assertRaises(EnhancerError):self.texture(data,'.dds')
    def test_cube_volume_array_and_custom_alpha_rejected(self):
        for offset,value in((24,3),(112,0x200),(112,0x200000),(28,99)):
            data=bytearray(dds_fixture());struct.pack_into('<I',data,offset,value)
            with self.subTest(offset=offset,value=value),self.assertRaises(EnhancerError):self.texture(data,'.dds')
        for dim,misc,array,alpha in((3,0,2,0),(3,4,1,0),(4,0,1,0),(3,0,1,4),(3,0,1,7)):
            data=bytearray(dds_fixture(dxgi=77));struct.pack_into('<5I',data,128,77,dim,misc,array,alpha)
            with self.subTest(values=(dim,misc,array,alpha)),self.assertRaises(EnhancerError):self.texture(data,'.dds')
    def test_unsupported_formats_rejected_before_roundtrip(self):
        for code in(95,96,98,99,81,84,2):
            data=bytearray(dds_fixture(dxgi=77));struct.pack_into('<I',data,128,code)
            with self.subTest(code=code),self.assertRaises(EnhancerError):self.texture(data,'.dds')
    def test_bad_masks_rejected(self):
        for masks in((255,255,255,0),(0xffff,0,0,0),(0x55,0xaa00,0xff0000,0)):
            data=bytearray(raw_dds());struct.pack_into('<4I',data,92,*masks)
            with self.subTest(masks=masks),self.assertRaises(EnhancerError):self.texture(data,'.dds')
    def test_bc1_cutout_alpha_survives(self):
        im=Image.new('RGBA',(8,8),(255,0,0,0));im.paste((0,255,0,255),(0,0,4,8));stream=io.BytesIO();im.save(stream,format='DDS',pixel_format='DXT1')
        tex,target,out=self.roundtrip(stream.getvalue(),'.dds')
        decoded=dds.decode(out,dds.inspect(out))
        self.assertEqual(decoded.getchannel('A').tobytes(),target.getchannel('A').tobytes())
    def test_tiny_odd_non_multiple_of_four(self):
        for size in((1,1),(1,7),(7,1),(5,3),(9,9)):
            for codec in('DXT1','DXT3','DXT5'):
                with self.subTest(size=size,codec=codec):self.roundtrip(dds_fixture(codec,size),'.dds')
    def test_independent_pillow_decoder_reads_legacy_outputs(self):
        for codec in('DXT1','DXT3','DXT5'):
            _,target,data=self.roundtrip(dds_fixture(codec),'.dds')
            with Image.open(io.BytesIO(data)) as decoded:
                decoded.load();self.assertEqual(decoded.size,target.size)


class TgaFormats(FormatCase):
    def test_color_depth_rle_and_all_origins(self):
        for bits in (15,16,24,32):
            for rle in(False,True):
                for origin in(0,16,32,48):
                    with self.subTest(bits=bits,rle=rle,origin=origin):
                        tex,target,out=self.roundtrip(tga_fixture(bits=bits,rle=rle,origin=origin),'.tga')
                        self.assertEqual(out[:12],tex.profile.details.header[:12])
                        self.assertEqual(out[16:18],tex.profile.details.header[16:18])
                        decoded=tga.decode(out,tga.inspect(out))
                        self.assertEqual(decoded.tobytes(),target.tobytes())
    def test_grayscale_alpha(self):
        for bits in(8,16):
            for rle in(False,True):
                tex,target,out=self.roundtrip(tga_fixture(bits=bits,rle=rle,gray=True),'.tga')
                self.assertEqual(tga.decode(out,tga.inspect(out)).tobytes(),target.tobytes())
    def test_indexed_palette_nonzero_first_and_16bit_indices(self):
        for bits,first in((8,0),(8,11),(16,256)):
            for rle in(False,True):
                tex,target,out=self.roundtrip(tga_fixture(bits=bits,rle=rle,indexed=True,first=first),'.tga')
                self.assertEqual(tga.decode(out,tga.inspect(out)).tobytes(),target.tobytes())
                self.assertEqual(tga.inspect(out).palette_bytes,tex.profile.details.palette_bytes)
    def test_two_and_four_way_interleaving(self):
        for interleave in(1,2):
            for origin in(0,16,32,48):
                tex,target,out=self.roundtrip(tga_fixture(interleave=interleave,origin=origin,rle=True),'.tga')
                self.assertEqual(tga.decode(out,tga.inspect(out)).tobytes(),target.tobytes())
    def test_extension_preserved(self):
        tex,target,out=self.roundtrip(tga_fixture(extension=True),'.tga')
        self.assertEqual(tga.inspect(out).extension,tex.profile.details.extension)
    def test_premultiplied_extension_preserved(self):
        data=bytearray(tga_fixture(extension=True));offset=struct.unpack_from('<I',data,len(data)-26)[0];data[offset+494]=4
        tex,target,out=self.roundtrip(data,'.tga')
        self.assertEqual(tga.inspect(out).extension[494],4)
    def test_rle_packet_boundaries(self):
        for width in(1,2,127,128,129,255,256,257):
            for color in('red',None):
                im=Image.new('RGBA',(width,3),color) if color else pattern((width,3))
                stream=io.BytesIO();im.save(stream,format='TGA',compression='tga_rle')
                tex,target,out=self.roundtrip(stream.getvalue(),'.tga')
                with Image.open(io.BytesIO(out)) as check:
                    self.assertEqual(check.convert('RGBA').tobytes(),target.tobytes())
    def test_missing_footer_stays_missing(self):
        data=tga_fixture()[:-26]
        _,_,out=self.roundtrip(data,'.tga')
        self.assertFalse(out.endswith(tga.FOOTER))
    def test_rle_overrun_truncation_and_invalid_indices(self):
        data=bytearray(tga_fixture(size=(1,1),rle=True));data[21]=255
        with self.assertRaises(EnhancerError):self.texture(data,'.tga')
        data=tga_fixture(rle=True)
        with self.assertRaises(EnhancerError):self.texture(data[:30],'.tga')
        data=bytearray(tga_fixture(indexed=True,bits=8));data[18+3+16]=255
        with self.assertRaises(EnhancerError):self.texture(data,'.tga')
    def test_private_developer_directory_refused(self):
        data=bytearray(tga_fixture());struct.pack_into('<I',data,len(data)-22,21)
        with self.assertRaises(EnhancerError):self.texture(data,'.tga')
    def test_independent_pillow_decoder_reads_rgba(self):
        _,target,out=self.roundtrip(tga_fixture(rle=True),'.tga')
        with Image.open(io.BytesIO(out)) as check:self.assertEqual(check.convert('RGBA').tobytes(),target.tobytes())


class PngFormats(FormatCase):
    def test_all_eight_bit_color_types_and_interlace(self):
        for color in(0,2,3,4,6):
            for interlace in(0,1):
                with self.subTest(color=color,interlace=interlace):
                    tex,target,out=self.roundtrip(png_fixture(color=color,interlace=interlace),'.png')
                    with Image.open(io.BytesIO(out)) as check:self.assertEqual(check.convert('RGBA').tobytes(),target.tobytes())
    def test_packed_grayscale_and_palette(self):
        for color in(0,3):
            for depth in(1,2,4):
                for interlace in(0,1):
                    for size in((1,1),(3,2),(17,11)):
                        with self.subTest(color=color,depth=depth,interlace=interlace,size=size):
                            tex,target,out=self.roundtrip(png_fixture(size,depth,color,interlace),'.png')
                            with Image.open(io.BytesIO(out)) as check:self.assertEqual(check.convert('RGBA').tobytes(),target.tobytes())
    def test_color_key_rgb_and_gray(self):
        for mode,key in(('L',40),('RGB',(40,60,80))):
            im=Image.new(mode,(13,7),key);im.putpixel((0,0),90 if mode=='L' else(90,80,70));stream=io.BytesIO();im.save(stream,format='PNG',transparency=key)
            tex,target,out=self.roundtrip(stream.getvalue(),'.png')
            self.assertEqual(png.inspect(out).transparency,tex.profile.details.transparency)
            with Image.open(io.BytesIO(out)) as check:self.assertEqual(check.convert('RGBA').tobytes(),target.tobytes())
    def test_rgba_storage_stays_rgba_when_opaque(self):
        im=Image.new('RGBA',(4,4),'red');stream=io.BytesIO();im.save(stream,format='PNG')
        tex,_,out=self.roundtrip(stream.getvalue(),'.png')
        self.assertEqual(out[25],6)
    def test_palette_alpha_not_ignored_when_rgb_duplicates(self):
        im=Image.new('P',(4,2),0);im.putpalette([255,0,0,255,0,0]+[0]*762);im.putpixel((0,0),1)
        stream=io.BytesIO();im.save(stream,format='PNG',transparency=bytes([255,0]))
        _,target,out=self.roundtrip(stream.getvalue(),'.png')
        with Image.open(io.BytesIO(out)) as check:self.assertEqual(check.convert('RGBA').tobytes(),target.tobytes())
    def test_crc_and_truncation(self):
        data=bytearray(png_fixture());data[29]^=1
        for bad in(data,png_fixture()[:-1],png_fixture()+b'extra'):
            with self.assertRaises(EnhancerError):self.texture(bad,'.png')
    def test_icc_retained_and_not_sent_with_pixels(self):
        icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        im=pattern();stream=io.BytesIO();im.save(stream,format='PNG',icc_profile=icc)
        tex,target,out=self.roundtrip(stream.getvalue(),'.png')
        self.assertEqual(tex.image.info,{})
        with Image.open(io.BytesIO(out)) as check:self.assertEqual(check.info['icc_profile'],icc)
    def test_unknown_critical_chunk_refused(self):
        data=png_fixture();data=data[:33]+png_chunk(b'ABCD',b'x')+data[33:]
        with self.assertRaises(EnhancerError):self.texture(data,'.png')


class JpegFormats(FormatCase):
    def test_modes_progressive_sampling_quantization(self):
        for mode in('L','RGB','CMYK'):
            for progressive in(False,True):
                for sampling in((0,1,2) if mode=='RGB' else(0,)):
                    with self.subTest(mode=mode,progressive=progressive,sampling=sampling):
                        im=pattern(opaque=True).convert(mode);stream=io.BytesIO();im.save(stream,format='JPEG',quality=86,subsampling=sampling,progressive=progressive)
                        tex,target,out=self.roundtrip(stream.getvalue(),'.JPEG')
                        with Image.open(io.BytesIO(out)) as check:
                            self.assertEqual(check.mode,mode);self.assertEqual(check.quantization,tex.profile.metadata['qtables'])
                            self.assertEqual(JpegImagePlugin.get_sampling(check),tex.profile.metadata['sampling'])
                            self.assertEqual(bool(check.info.get('progressive')),progressive)
    def test_exif_orientation_and_dimensions(self):
        image=pattern((17,11),opaque=True).convert('RGB');exif=image.getexif();exif[274]=6;exif[256]=17;exif[257]=11
        exif[34665]={40962:17,40963:11}
        stream=io.BytesIO();image.save(stream,format='JPEG',exif=exif)
        tex,target,out=self.roundtrip(stream.getvalue(),'.jpg')
        self.assertEqual(tex.image.size,(17,11))
        with Image.open(io.BytesIO(out)) as check:
            self.assertEqual(check.getexif()[274],6);self.assertEqual(check.getexif()[256],34);self.assertEqual(check.getexif()[257],22)
            self.assertEqual(check.getexif().get_ifd(34665)[40962],34)
    def test_icc_and_comment_retained(self):
        icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        stream=io.BytesIO();pattern(opaque=True).convert('RGB').save(stream,format='JPEG',icc_profile=icc,comment=b'asset note')
        _,_,out=self.roundtrip(stream.getvalue(),'.jpg')
        with Image.open(io.BytesIO(out)) as check:
            self.assertEqual(check.info['icc_profile'],icc);self.assertEqual(check.info['comment'],b'asset note')
    def test_extension_mismatch_refused(self):
        stream=io.BytesIO();pattern().save(stream,format='PNG')
        with self.assertRaises(EnhancerError):self.texture(stream.getvalue(),'.jpg')


class NativeStorage(FormatCase):
    def test_source_protected_and_filename_type_checked(self):
        texture=self.texture(dds_fixture(),'.dds');original=texture.path.read_bytes()
        with self.assertRaises(EnhancerError):save_original_atomic(texture.image,texture.profile,texture.path,texture.path,overwrite=True)
        with self.assertRaises(EnhancerError):save_original_atomic(texture.image,texture.profile,self.root/'not.dds.png')
        self.assertEqual(texture.path.read_bytes(),original)
    def test_encode_failure_never_replaces_output(self):
        texture=self.texture(png_fixture(),'.png');destination=self.root/'output.png';destination.write_bytes(b'old')
        with patch('enhancer.storage.encode_original',side_effect=EnhancerError('bad')),self.assertRaises(EnhancerError):
            save_original_atomic(texture.image,texture.profile,destination,overwrite=True)
        self.assertEqual(destination.read_bytes(),b'old')
    def test_native_output_has_correct_magic(self):
        for source,ext,magic in((dds_fixture(),'.dds',b'DDS '),(png_fixture(),'.png',b'\x89PNG')):
            tex=self.texture(source,ext);dest=self.root/('native'+ext)
            save_original_atomic(tex.image,tex.profile,dest,tex.path)
            self.assertTrue(dest.read_bytes().startswith(magic))

if __name__=='__main__':unittest.main()
