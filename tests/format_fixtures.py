"""Synthetic, redistributable fixtures with explicit headers and corner patterns."""
from __future__ import annotations
import io
import struct
import zlib
import numpy as np
from PIL import Image


def pattern(size=(17, 11), opaque=False):
    w,h=size
    y,x=np.indices((h,w))
    rgba=np.stack(((x*31+y*7)%256,(x*11+y*47)%256,(x*71+y*3)%256,
                   np.full((h,w),255) if opaque else (x*29+y*13)%256),axis=2).astype('uint8')
    return Image.fromarray(rgba)


def dds_fixture(codec='DXT5',size=(16,8),levels=1,dxgi=None,premul=False):
    image=pattern(size)
    if premul:
        a=np.asarray(image).astype('uint16').copy();a[:,:,:3]=(a[:,:,:3]*a[:,:,3:4]+127)//255
        image=Image.fromarray(a.astype('uint8'))
    pieces=[]
    header=None
    real={'DXT2':'DXT3','DXT4':'DXT5'}.get(codec,codec)
    for i in range(levels):
        sub=image.resize((max(1,size[0]>>i),max(1,size[1]>>i)))
        stream=io.BytesIO();sub.save(stream,format='DDS',pixel_format=real)
        raw=stream.getvalue();header=header or bytearray(raw[:128]);pieces.append(raw[128:])
    struct.pack_into('<I',header,28,levels)
    flags=struct.unpack_from('<I',header,8)[0]
    if levels>1:flags|=0x20000
    struct.pack_into('<I',header,8,flags)
    struct.pack_into('<I',header,108,0x1000|(0x400008 if levels>1 else 0))
    header[84:88]=codec.encode()
    if dxgi is not None:
        header[84:88]=b'DX10';header.extend(struct.pack('<5I',dxgi,3,0,1,2 if premul else 1))
    return bytes(header)+b''.join(pieces)


def raw_dds(size=(7,5),bits=32,masks=(0xff0000,0xff00,0xff,0xff000000),kind='rgb',alignment=1,levels=1):
    image=pattern(size)
    header=bytearray(128);header[:4]=b'DDS '
    pitch=((size[0]*(bits//8)+alignment-1)//alignment)*alignment
    struct.pack_into('<7I',header,4,124,0x100f|(0x20000 if levels>1 else 0),size[1],size[0],pitch,0,levels)
    pflags={'rgb':0x40,'luminance':0x20000,'alpha':2}[kind]
    if masks[3] and kind!='alpha':pflags|=1
    struct.pack_into('<8I',header,76,32,pflags,0,bits,*masks)
    struct.pack_into('<I',header,108,0x1000|(0x400008 if levels>1 else 0))
    output=bytearray(header)
    for i in range(levels):
        sub=image.resize((max(1,size[0]>>i),max(1,size[1]>>i)))
        for y in range(sub.height):
            row=bytearray()
            for x in range(sub.width):
                pixel=sub.getpixel((x,y));value=0
                if kind=='luminance':pixel=(sub.convert('L').getpixel((x,y)),)*3+(pixel[3],)
                for c,mask in enumerate(masks):
                    if mask and not(kind=='luminance' and c in (1,2)):
                        shift=(mask & -mask).bit_length()-1;maximum=mask>>shift
                        value|=((pixel[c]*maximum+127)//255)<<shift
                row.extend(value.to_bytes(bits//8,'little'))
            row.extend(b'\0'*((-len(row))%alignment));output.extend(row)
    return bytes(output)


def tga_fixture(size=(17,11),bits=32,rle=False,origin=0x20,gray=False,indexed=False,first=0,interleave=0,extension=False):
    image=pattern(size,opaque=bits==24)
    kind=1 if indexed else (3 if gray else 2)
    palette=b''
    attributes=8 if bits==32 or(gray and bits==16) else (1 if bits==16 and not indexed else 0)
    header=struct.pack('<BBBHHBHHHHBB',3,int(indexed),kind+(8 if rle else 0),first,4 if indexed else 0,32 if indexed else 0,
                       7,9,*size,bits,origin|attributes|(interleave<<6))
    if indexed:
        palette=bytes((0,0,255,255,0,255,0,128,255,0,0,0,30,60,90,255))
    rows=[]
    for y in range(size[1]):
        values=[]
        sy=y if origin & 32 else size[1]-1-y
        for x in range(size[0]):
            sx=size[0]-1-x if origin & 16 else x
            r,g,b,a=image.getpixel((sx,sy))
            if indexed:pixel=((sx+sy)%4+first).to_bytes(bits//8,'little')
            elif gray:pixel=bytes([r,a]) if bits==16 else bytes([r])
            elif bits in(15,16):pixel=((r*31//255)<<10|(g*31//255)<<5|(b*31//255)|((int(a>=128)<<15) if bits==16 else 0)).to_bytes(2,'little')
            elif bits==24:pixel=bytes((b,g,r))
            else:pixel=bytes((b,g,r,a))
            values.append(pixel)
        rows.append(values)
    order=[y for start in range(1<<interleave) for y in range(start,size[1],1<<interleave)]
    payload=bytearray()
    for y in order:
        values=rows[y]
        if rle:
            # independent encoder uses only raw packets, still valid RLE input
            for start in range(0,len(values),128):
                part=values[start:start+128];payload.append(len(part)-1);payload.extend(b''.join(part))
        else:payload.extend(b''.join(values))
    result=bytearray(header+b'id!'+palette+payload)
    ext_offset=0
    if extension:
        ext_offset=len(result);ext=bytearray(495);struct.pack_into('<H',ext,0,495);ext[494]=3;result.extend(ext)
    result.extend(struct.pack('<II',ext_offset,0)+b'TRUEVISION-XFILE.\0')
    return bytes(result)


def png_chunk(kind,data):
    return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)


def png_fixture(size=(17,11),depth=8,color=6,interlace=0):
    # independent raw-row generator, including Adam7 and packed samples
    channels={0:1,2:3,3:1,4:2,6:4}[color]
    maximum=(1<<depth)-1
    y,x,c=np.indices((size[1],size[0],channels))
    samples=((x*3+y*5+c*11)%(maximum+1)).astype('uint8')
    result=b'\x89PNG\r\n\x1a\n'+png_chunk(b'IHDR',struct.pack('>IIBBBBB',*size,depth,color,0,0,interlace))
    if color==3:
        count=1<<depth
        palette=bytes(v for i in range(count) for v in(i*255//max(1,count-1),i*33%256,i*17%256))
        result+=png_chunk(b'PLTE',palette)+png_chunk(b'tRNS',bytes(i*255//max(1,count-1) for i in range(count)))
    raw=bytearray()
    passes=[(0,0,8,8),(4,0,8,8),(0,4,4,8),(2,0,4,4),(0,2,2,4),(1,0,2,2),(0,1,1,2)] if interlace else [(0,0,1,1)]
    for xx,yy,dx,dy in passes:
        for y in range(yy,size[1],dy):
            row=samples[y,xx::dx]
            if len(row)==0:continue
            raw.append(0)
            if depth==8:raw.extend(row.tobytes())
            else:
                value=0;used=0
                for sample in row[:,0]:
                    value=(value<<depth)|int(sample);used+=depth
                    if used==8:raw.append(value);value=used=0
                if used:raw.append(value<<(8-used))
    return result+png_chunk(b'IDAT',zlib.compress(bytes(raw)))+png_chunk(b'IEND',b'')
