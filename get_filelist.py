from pathlib import Path
import typing as tp
import os

def get_filelist(
    folder: tp.Union[str, os.PathLike],
    extensions: tp.Optional[tp.List[str]] = None
) -> tp.List[str]:
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a',".opus"]
    extensions = set(ext.lower() for ext in extensions)
    
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"The provided directory '{folder}' is not a valid directory.")
    filelist = [str(file) for file in path.rglob('*') if file.suffix.lower() in extensions]
    return filelist

def get_filelist(
    folder: tp.Union[str, os.PathLike],
    extensions: tp.Optional[tp.List[str]] = None
) -> tp.Generator[str, None, None]:
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a',".opus"]
    extensions = set(ext.lower() for ext in extensions)
    
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"The provided directory '{folder}' is not a valid directory.")
    
    return (str(file) for file in path.rglob('*') if file.suffix.lower() in extensions)


if __name__ == "__main__":

    AUDIO_FOLDER = "raw_audio"
    OUTPUT_TXT = "configs/data/youtube.jsonl"



    filelist = get_filelist(AUDIO_FOLDER)

    '''
        two input in meta
        c : auther
        ct: transcript
    
    '''
    import json
    with open(OUTPUT_TXT, 'w') as f:
        for file_path in filelist:

            # father_folder = Path(file_path).parent.name.split(" ")[0]
            # author = Path(file_path).parent.name.split(" ")[0]

            author = Path(file_path).parent.name
            # import pdb;pdb.set_trace()
            f.write(json.dumps(
                {
                    "path": file_path,
                    "author":author ,
                    }, ensure_ascii=False
                ) + "\n"
            )

"""
    output:
    {"path": "aaa.wav", "author": "Joseph Haydn"}
    {"path": "bbb.wav", "author": "Joseph Haydn"}

"""



