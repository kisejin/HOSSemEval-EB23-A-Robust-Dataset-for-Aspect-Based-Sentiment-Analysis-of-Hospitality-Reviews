import json
import pandas as pd
import argparse
import logging

logger = logging.getLogger(__name__)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test size')
    
    return parser.parse_args()


polarities = ['positive', 'negative', 'neutral']

def extract_info_nest_data(rows: dict) -> pd.DataFrame:

    annotations = []
    results = []
    for row in rows:
        # Get id user
        id_user = {'id_user': row['id']}

        # Get raw sentiment of user
        text = row['data']['text']

        # Keep user id, text and sentiment position
        for annotation in row['annotations']:
            result = {'text': text,'result': annotation['result']}
            results.append({**id_user,**result})

    # Previous sentiment positions are get all labels
    start_prev, end_prev = 0, 0
    text_prev = ''
    result_values = []

    # Loop through each row
    for i, result in enumerate(results):
        values = result['result']

        # v is list of triplet (aspect term, aspect category, sentiment polarity) in a sentence
        # v_i is each position in triplet
        v, v_i = [], []

        # Loop through each position in raw text
        for value in values:
            value = value['value']

            # Get the current position
            start, end = value['start'], value['end']

            # Check if the previous position is the same as current, we add the quadlet into list,
            #  otherwise we create a new quadlet with aspect term, aspect category and update previous step to current
            if start == start_prev and end == end_prev and text_prev == value['text']:
                v_i.append(value['labels'][0].lower())
                if v_i[1].lower() in polarities:
                    tmp = v_i[1]
                    v_i[1] = v_i[2]
                    v_i[2] = tmp

                v.append(v_i)
                v_i = []
            else:
                if len(v_i) >= 2:
                    if v_i[-1] not in polarities:
                        polar = 'neutral'
                        v_i.append(polar)
                        v.append(v_i)
                    v_i = []
                v_i.append(value['text'].lower())
                v_i.append(value['labels'][0].lower())
                start_prev, end_prev = start, end
                text_prev = value['text']

        text_prev = ''
        results[i]['result'] = v


    return pd.DataFrame(results)


def preprocess_data(file_path: str):
    data = json.loads(open(file_path))
    return extract_info_nest_data(data)

def write_data(df: pd.DataFrame, output_path: str):
    with open(output_path, "w") as file:
        for i, row in df.iterrows(): # pandas
        # for row in df.rows(named=True): # polars
            # text = [tuple(v) for v in row['result']]
            output_string = f"{row['text']} ####{row['result']}\n"
            # output_string = f"{row['user_id']},{row['text']} ####{row['result']}\n"
            file.write(output_string)

    print(f"Content written to {output_path}")
    
def split_data(df: pd.DataFrame, test_size, output_path: str):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_size, random_state=0)
    list_file = ['train.txt', 'dev.txt', 'test.txt']
    for data in list_file:
        if 'train' in data:
            write_data(train, f"{output_path}/{data}")
        else:
            write_data(test, f"{output_path}/{data}")
    

if __name__ == "__main__":
    
    args = init_args()
    df = preprocess_data(args.input)
    split_data(df, args.test_size, args.output)
    
    
    