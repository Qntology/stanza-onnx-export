import stanza
def check_pretrain_root(lang):
    nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse', download_method=None, device='cpu')
    m = nlp.processors['depparse'].trainer.model
    print(f"pretrain_root: {m.args.get('pretrain_root')}")
    print(f"use_upos: {m.args.get('use_upos')}")
    print(f"use_xpos: {m.args.get('use_xpos')}")
    print(f"use_ufeats: {m.args.get('use_ufeats')}")

if __name__ == "__main__":
    check_pretrain_root('ko')
