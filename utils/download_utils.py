import os
from pathlib import Path
import sys
import tqdm
import requests
import validators


class FileDownloader(object):
    @staticmethod
    def get_url_filename(url):
        r"""
        Discover file name from HTTP URL, If none is discovered derive name from http redirect HTTP content header Location
        :param url: Url link to file to download
        :type url: str
        :return: Base filename
        :rtype: str
        """
        try:
            if not validators.url(url):
                raise ValueError('Invalid url')
            filename = os.path.basename(url)
            basename, ext = os.path.splitext(filename)
            if ext:
                return filename
            header = requests.head(url, allow_redirects=False).headers
            return os.path.basename(header.get('Location')) if 'Location' in header else filename
        except requests.exceptions.HTTPError as errh:
            print("Http Error:", errh)
            raise errh
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:", errc)
            raise errc
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:", errt)
            raise errt
        except requests.exceptions.RequestException as err:
            print("OOps: Something Else", err)
            raise err

    @staticmethod
    def download_file(url, filename=None, target_dir=None):
        r"""
        Stream downloads files via HTTP
        :param url: Url link to file to download
        :type url: str
        :param filename: filename overrides filename defined in Url param
        :type filename: str
        :param target_dir: target destination directory to download file to
        :type target_dir: str
        :return: Absolute path to target destination where file has been downloaded to
        :rtype: str
        """
        if target_dir and not os.path.isdir(target_dir):
            raise ValueError('Invalid target_dir={} specified'.format(target_dir))
        local_filename = self.get_url_filename(url) if not filename else filename

        req = requests.get(url, stream=True)
        file_size = int(req.headers['Content-Length'])
        chunk_size = 1024  # 1 MB
        num_bars = int(file_size / chunk_size)

        base_path = os.path.abspath(os.path.dirname(__file__))
        target_dest_dir = os.path.join(base_path, local_filename) if not target_dir else os.path.join(target_dir,
                                                                                                      local_filename)
        with open(target_dest_dir, 'wb') as fp:
            for chunk in tqdm.tqdm(req.iter_content(chunk_size=chunk_size), total=num_bars, unit='KB',
                                   desc=local_filename, leave=True, file=sys.stdout):
                fp.write(chunk)

        return target_dest_dir


'''
if __name__ == "__main__":

    links = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]

    downloader = FileDownloader()

    for url in links:
        tgt_dir = str(Path(os.path.abspath(os.path.dirname(__file__))).parent.joinpath('data').joinpath('raw'))
        downloader.download_file(url, filename=None, target_dir=tgt_dir)
'''
