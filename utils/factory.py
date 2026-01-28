from methods.dsca import DSCA

def get_model(model_name, args):
    name = model_name.lower()
    options = {
               'dsca': DSCA,
               }
    return options[name](args)

