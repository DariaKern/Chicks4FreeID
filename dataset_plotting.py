
from datasets import load_dataset, Image, DownloadMode
import plotly.express as px # Used for creating the plot
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    dataset = load_dataset(
        "dariakern/Chicks4FreeID", 
        "full-dataset", 
        split="train",
        download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS
    )

    def extract_label_names(item):
        # Create a new dictionary for the transformed batch
        new_item = {}
        
        # Transform instances
        new_instances = []
        for instance in item["instances"]:
            new_instance = {}
            features = dataset.features["instances"][0]
            # Convert class label indices to names
            new_instance["animal_category"] = features["animal_category"].int2str(instance["animal_category"])
            new_instance["identity"] = features["identity"].int2str(instance["identity"])
            new_instance["visibility"] = features["visibility"].int2str(instance["visibility"])
            new_instances.append(new_instance)
        
        # Add transformed instances to the new batch
        new_item["instances"] = new_instances
        new_item["coop"] = dataset.features["coop"].int2str(item["coop"])
        return new_item

    result = [extract_label_names(item) for item in tqdm(dataset.to_iterable_dataset())]

    instance_info = [
        {
            'Image': i,
            'Instance': instance_id,
            'Animal Category': instance['animal_category'],
            'Identity': instance['identity'],
            'Visibility': instance['visibility']
        }
        for i, entry in enumerate(result)
        for instance_id, instance in enumerate(entry['instances'])
    ]


    # Create a DataFrame
    df = pd.DataFrame(instance_info)

    ducks_and_roosters = [
        "Elvis", "Jackson", "Marley", "Evelyn"
    ]

    # Group by 'Identity' and 'Visibility' and count the occurrences
    counts_df = df.groupby(['Identity', 'Visibility']).size().unstack(fill_value=0).reset_index()
    counts_df = counts_df[counts_df.Identity != "Unknown"]
    counts_df["Identity"] = counts_df.Identity.map(lambda x: x if x not in ducks_and_roosters else f"*{x}")
    counts_df = counts_df.sort_values("best", ascending=False)

    # Convert wide to long format for plotting
    counts_df = counts_df.melt(id_vars='Identity', value_vars=['good', 'best', 'bad'], var_name='Visibility', value_name='Count')

    fig = px.bar(
        counts_df,
        x = "Identity",
        y = "Count",
        color ="Visibility",
        #orientation="v",
        category_orders = {"Visibility": ["best", "good", "bad"]},
        color_discrete_sequence= ['#1f77b4', '#aec7e8', '#4c78a8'] ,
        pattern_shape="Visibility"
    )

    fig.update_layout(
        height=450,
        width=1000,
        font_family="Georgia",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.975
        )
    )

    fig.show()
    fig.write_image("overview.pdf")