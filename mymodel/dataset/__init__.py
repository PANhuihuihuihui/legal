from legal_element import get_element_datasets,get_scm_datasets

def build_datasets(args):
    if args.stage ==1 :
        return get_element_datasets(args)
    else:
        return get_scm_datasets