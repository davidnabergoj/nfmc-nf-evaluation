from collections import defaultdict

import json
from multiprocessing import Pool

from pathlib import Path


def load_data(file_path):
    with open(str(file_path), 'r') as _f:
        try:
            return json.load(_f)
        except json.JSONDecodeError:
            print(f'Error parsing json file {file_path}')
            return {}


if __name__ == '__main__':
    results_directories = filter(lambda p: p.is_dir(), Path(__file__).parent.rglob('results'))

    data = defaultdict(list)
    pool = Pool()
    for d in results_directories:
        print(f'Processing {d}')
        d: Path
        files = d.glob('*.json')
        for file_data in pool.imap(load_data, files):
            data[str(d.parent.stem)].append(file_data)
    pool.close()
    pool.join()

    with open('results.json', 'w') as output_file:
        json.dump(data, output_file)
