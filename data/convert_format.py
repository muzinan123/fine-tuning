import json

tools = [{
    "name": "search_hotels",
    "description": "Generate search criteria based on user requirements to search for hotels",
    "parameters": {
        "type": "object",
        "properties": {
            "name": { "type": "string", "description": "Hotel name" },
            "type": { "type": "string", "enum": ["Luxury", "Economy", "Comfort", "High-end"], "description": "Hotel type" },
            "facilities": { "type": "array", "items": { "type": "string" }, "description": "List of facilities provided by the hotel" },
            "price_range_lower": { "type": "number", "minimum": 0, "description": "Lower price limit" },
            "price_range_upper": { "type": "number", "minimum": 0, "description": "Upper price limit" },
            "rating_range_lower": { "type": "number", "minimum": 0, "maximum": 5, "description": "Lower rating limit" },
            "rating_range_upper": { "type": "number", "minimum": 0, "maximum": 5, "description": "Upper rating limit" }
    }, "required": [] }
}]
tool_description = json.dumps(tools, ensure_ascii=False)

def read_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json_str = json.dumps(item,ensure_ascii=False)
            f.write(json_str + '\n')

def is_subset(sub_list, main_list):
    return all(item in main_list for item in sub_list)

def filter_subsets(lst):
    parsed_contexts = [(item, json.loads(item['context'])) for item in lst]
    return [item for item, context in parsed_contexts if not any(
        is_subset(context, json.loads(main_item['context'])) and item != main_item
        for main_item in lst)]

def convert(input_filename, output_filename):
    dataset = []
    lines = filter_subsets(read_jsonl(input_filename))
    for line in lines:
        data = {"tools":[tool_description.strip()],"conversations":[]}
        dialog = []
        dialog.extend(eval(line['context']))
        dialog.append(eval(line['response']))
        for turn in dialog:
            if turn["role"] == "search":
                think = {"role":"assistant","content":"I need to use the search_hotels tool to query hotels"}
                data["conversations"].append(think)
                action = {"role":"tool","name":"search_hotels","parameters":turn["arguments"]}
                data["conversations"].append(action)
            elif turn["role"] == "return":
                data["conversations"][-1]["observation"] = turn["records"]
            else:
                data["conversations"].append(turn)
        dataset.append(data)
    write_jsonl(dataset, output_filename)

if __name__ == '__main__':
    convert('train.llama2.jsonl', 'train.chatglm3.jsonl')
    convert('dev.llama2.jsonl', 'dev.chatglm3.jsonl')
    convert('test.llama2.jsonl', 'test.chatglm3.jsonl')
