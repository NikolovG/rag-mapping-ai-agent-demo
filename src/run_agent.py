from api_runner import index_yaml, suggest_mappings, apply_decisions

# 1) Build index
res = index_yaml("./mappings")
print("Index:", res["result"])


# 2) Generate suggestions
def res2():
    res2 = suggest_mappings("../raw.csv")
    print("Suggestions:", res2["result"].keys())
    print("Log:", res2["log"])

# 3) If human review needed, apply decisions later
# if res2["log"] == "waiting_for_decision":
#     rid = res2["result"]["review_id"]
#     print("Waiting for decision:", rid)

#     # after placing the decision file in review_queue/<rid>.decision.json
#     res3 = apply_decisions(rid)
#     print("Post-validation:", res3["result"]["post_validation"])
