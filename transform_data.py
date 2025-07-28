# load data C:\Users\paul-\Documents\Coding\CCI\data\collection-24_UnitedWayDane_nv-embed_processed_output.pkl
import pandas as pd

data = pd.read_pickle(
    r"C:\Users\paul-\Documents\Coding\CCI\data\collection-24_UnitedWayDane_nv-embed_processed_output.pkl"
)
# print columns
print(data.columns)
REQUIRED_COLUMNS = ["dialogue_id", "turn_index", "speaker", "text", "embedding"]
# rename "Conversation ID" to "dialogue_id", Speaker Name to "speaker", and Content to "text", and Latent-Attention_Embedding to embedding
data = data.rename(
    columns={
        "Conversation ID": "dialogue_id",
        "Speaker Name": "speaker",
        "Content": "text",
        "Latent-Attention_Embedding": "embedding",
        "Index in Conversation": "turn_index"
    }
)
# drop all other columns
data = data[REQUIRED_COLUMNS]
# save data as data.pkl
data.to_pickle(r"C:\Users\paul-\Documents\Coding\CCI\data\data.pkl")