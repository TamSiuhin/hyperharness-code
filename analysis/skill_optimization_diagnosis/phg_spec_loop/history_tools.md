- {"action": "search", "query": "...", "filters": {...}, "limit": 1-20} ranks this user's history events by BM25 and returns
  ids, day, app, interaction, self_authored, dm and a short snippet. Use specific words; search what the digest omits.
- {"action": "list", "filters": {...}, "order": "recent" | "oldest", "limit": 1-30} lists events matching the filters.
- {"action": "read", "ids": ["..."]} returns up to 5 complete events (long ones truncated).
  Filters (all optional): "app" (string or list), "interaction" (for example "explicit_positive", "implicit_negative"),
  "self_authored" (true/false), "dm" (true/false), "since" and "until" ("YYYY-MM-DD"). Tool results are truncated and
  have their own call budget. The digest is a lossy summary: verify important claims, look for counterexamples and
  recent changes, and read the user's own writing before encoding voice.
