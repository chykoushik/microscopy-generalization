from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/downloads/missing_marker'
OUT = ROOT / 'outputs/results/missing_marker'

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < 20 * 1024 ** 3:
        raise RuntimeError('Storage reserve')
    sources = [('platemap.txt', 'https://data.broadinstitute.org/bbbc/BBBC013/BBBC013_v1_platemap_all.txt', 100000), ('images.zip', 'https://data.broadinstitute.org/bbbc/BBBC013/BBBC013_v1_images_bmp.zip', 80000000)]
    ledger = []
    for name, url, limit in sources:
        path = CACHE / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.part')
            downloaded = temporary.stat().st_size if temporary.exists() else 0
            request = urllib.request.Request(url, headers={'Range': f'bytes={downloaded}-'} if downloaded else {})
            with urllib.request.urlopen(request, timeout=90) as response:
                append = response.status == 206 and downloaded > 0
                with temporary.open('ab' if append else 'wb') as stream:
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        stream.write(chunk)
                        if stream.tell() > limit:
                            raise RuntimeError('Download cap')
            temporary.replace(path)
        ledger.append(dict(path=str(path.relative_to(ROOT)), url=url, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), license='CC-BY-3.0 Ilya Ravkin', channel_definition='C1 FKHR-GFP held-out marker; C2 DRAQ DNA input', units='96 physical wells; four replicate rows per drug; one plate'))
    with zipfile.ZipFile(CACHE / 'images.zip') as archive:
        names = archive.namelist()
        if sum((v.file_size for v in archive.infolist())) > 500000000:
            raise RuntimeError('Expansion cap')
        dest = CACHE / 'images'
        for name in names:
            target = (dest / name).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError('Unsafe archive')
            if not target.exists():
                archive.extract(name, dest)
    (OUT / 'DOWNLOAD_LEDGER.json').write_text(json.dumps(ledger, indent=2))
    (OUT / 'SECOND_DATASET_ELIGIBILITY.md').write_text('BBBC013 official source confirms DNA channel2 and held-out FKHR-GFP channel1,96 physical wells, four replicate rows per drug,640x640 images and CC-BY3.0. A bounded31MB ZIP is reused/downloaded with caps. Cell labels will be explicitly derived nuclear/ring marker ratios, not claimed provided expert annotations. DNA segmentation and all morphology inputs exclude GFP. One plate gives independent wells but no independent batch replication. Source: https://bbbc.broadinstitute.org/BBBC013\n')
    print('Acquisition complete')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/results/missing_marker')
    OUT = ROOT / parser.parse_args().output
    main()
