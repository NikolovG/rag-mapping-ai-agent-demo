rag_mapper.py  →  handles model + retrieval logic
langgraph_rag_agent.py  →  orchestrates workflow and CLI
 

Instructions for usage 

### Building the index 
python3 langgraph_rag_agent.py index --yaml_dir ./mappings --model ./rag_index.npz

### Testing mapping suggestion functionality
python3 langgraph_rag_agent.py suggest --model ./rag_index.npz --csv ../raw.csv --k 5

### Merge additional YAML file into training set
python langgraph_rag_agent.py merge --yaml_dir ./new_mappings --model rag_index.npz

Expected output
{
  "Client": [
    {"target": "customer_name", "score": 0.94, "retrieval_sim": 0.89, "clf_prob": 0.80}
  ],
  "Order Number": [
    {"target": "order_id", "score": 0.91, "retrieval_sim": 0.87, "clf_prob": 0.77}
  ],
  "Purchase Date": [
    {"target": "order_date", "score": 0.89, "retrieval_sim": 0.86, "clf_prob": 0.70}
  ],
  "Amount": [
    {"target": "total_amount", "score": 0.93, "retrieval_sim": 0.90, "clf_prob": 0.79}
  ],
  "Item": [
    {"target": "product_name", "score": 0.88, "retrieval_sim": 0.84, "clf_prob": 0.68}
  ]
}

