import torch
import cytoolz as tz


def load_models(langs: str,
                torchhub_dir: str = None,
                device: str = 'cuda'):
    if torchhub_dir:
        # Set the directory where the models will be stored.
        torch.hub.set_dir(torchhub_dir)

    if langs == 'en2de':
        # Round-trip translations between English and German
        src2tgt = torch.hub.load('pytorch/fairseq',
                                 'transformer.wmt19.en-de.single_model',
                                 tokenizer='moses',
                                 bpe='fastbpe')
        tgt2src = torch.hub.load('pytorch/fairseq',
                                 'transformer.wmt19.de-en.single_model',
                                 tokenizer='moses',
                                 bpe='fastbpe')

    elif langs == 'en2ru':
        # Round-trip translations between English and Russian
        src2tgt = torch.hub.load('pytorch/fairseq',
                                 'transformer.wmt19.en-ru.single_model',
                                 tokenizer='moses',
                                 bpe='fastbpe')
        tgt2src = torch.hub.load('pytorch/fairseq',
                                 'transformer.wmt19.ru-en.single_model',
                                 tokenizer='moses',
                                 bpe='fastbpe')
    else:
        raise NotImplementedError

    return src2tgt.to(device), tgt2src.to(device)


def batch_backtranslation(src_sentences,
                          src2tgt,
                          tgt2src,
                          n_src2tgt=5,
                          src2tgt_topk=100,
                          src2tgt_temp=1.0,
                          n_tgt2src=5,
                          tgt2src_topk=100,
                          tgt2src_temp=1.0,
                          ):
    """
    Perform backtranslation using the fairseq pretrained translation models.
    src2tgt and tgt2src are assumed to be fairseq torch models.
    """
    # Half precision
    src2tgt = src2tgt.half()
    tgt2src = tgt2src.half()

    # Encode the source sentence
    src_sentences_bin = [src2tgt.encode(e)[:1024] for e in src_sentences]

    # Translate it
    tgt_sentences = src2tgt.generate(src_sentences_bin,
                                     beam=n_src2tgt,
                                     sampling=True,
                                     sampling_topk=src2tgt_topk,
                                     temperature=src2tgt_temp,
                                     skip_invalid_size_inputs=True,
                                     )

    # Back-translate: moving tokens to CPU because of an error otherwise
    src_paraphrases = tgt2src.generate([e['tokens'].cpu() for l in tgt_sentences for e in l],
                                       beam=n_tgt2src,
                                       sampling=True,
                                       sampling_topk=tgt2src_topk,
                                       temperature=tgt2src_temp,
                                       skip_invalid_size_inputs=True,
                                       )

    # Flatten out all the translations into one giant list
    flat_src_paraphrases = list(
        tz.concat(map(lambda l: list(map(lambda e: tgt2src.decode(e['tokens']), l)), src_paraphrases))
    )

    # Partition so that we get n_src2tgt * n_tgt2src paraphrases per input sentence
    return list(tz.partition_all(len(flat_src_paraphrases) // len(src_sentences), flat_src_paraphrases))
