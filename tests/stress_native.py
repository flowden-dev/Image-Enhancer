"""Optional large-output smoke test; no network or API credentials are used."""
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from enhancer.api import run_job
from enhancer.core import Settings, load_texture
from enhancer.formats import decode_source
from format_fixtures import dds_fixture, tga_fixture, pattern


def main():
    with tempfile.TemporaryDirectory(prefix='image-enhancer-stress-') as directory:
        root = Path(directory)
        for edge in (2048, 4096):
            for extension in ('.dds', '.tga', '.png', '.jpg'):
                source = root / ('source' + extension)
                if extension == '.dds':
                    source.write_bytes(dds_fixture('DXT5', (64, 64), levels=7))
                elif extension == '.tga':
                    source.write_bytes(tga_fixture((64, 64), rle=True))
                elif extension == '.jpg':
                    pattern((64, 64)).convert('RGB').save(source, quality=91, progressive=True)
                else:
                    pattern((64, 64)).save(source)
                original = source.read_bytes()
                started = time.perf_counter()
                texture = load_texture(source)
                result = run_job(texture, Settings(method='lanczos', target_edge=edge), '', root / 'outputs')
                assert result.image.size == (edge, edge)
                assert result.original_path is not None and result.original_path.exists(), result.warnings
                assert not result.warnings, result.warnings
                assert source.read_bytes() == original
                print(f'{edge} x {edge} {extension}: validated native export, source unchanged; {time.perf_counter() - started:.2f}s', flush=True)
                result.image.close()
                texture.image.close()
                del result, texture
        if resource is not None and sys.platform.startswith('linux'):
            print(f'Peak resident memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MiB (Linux)', flush=True)


if __name__ == '__main__':
    main()
