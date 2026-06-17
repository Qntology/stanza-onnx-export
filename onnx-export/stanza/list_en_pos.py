import stanza
import json

resources = stanza.resources.common.load_resources_json()
en_models = resources.get('en', {})
pos_models = en_models.get('pos', {})
print(json.dumps(list(pos_models.keys()), indent=2))
