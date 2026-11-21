"""Task-separated construction evidence, adapted from historical TRAIN v5."""
import json


def digest(events):
    buckets = {}
    categories = {}
    products = {}
    review_lengths = {}
    titles = {}
    rating_counts = {}
    domains = {}
    title_lengths = []
    title_bins = {"1-2": 0, "3-5": 0, "6-10": 0, ">10": 0}
    ratings = []
    for event in events:
        details = event["event"]
        domain = str(details.get("domain") or event.get("app") or "unknown")[:120]
        domains[domain] = domains.get(domain, 0) + 1
        title = str(details.get("review_title") or "").strip()
        if title:
            length = len(title.split())
            title_lengths.append(length)
            group = "1-2" if length <= 2 else "3-5" if length <= 5 else "6-10" if length <= 10 else ">10"
            title_bins[group] += 1
        rating = details.get("rating")
        if type(rating) not in (int, float) or not 1 <= rating <= 5:
            continue
        ratings.append(rating)
        key = str(float(rating))
        rating_counts[key] = rating_counts.get(key, 0) + 1
        buckets.setdefault((domain, rating), []).append(event)
        product = details.get("product") or {}
        category = str(product.get("category") or "other")[:160]
        categories.setdefault((domain, category), []).append(rating)
        if product.get("title"):
            products.setdefault((domain, rating), []).append(
                {"id": event["id"], "title": str(product["title"])[:110]})
        review = str(details.get("review") or "")
        if review.strip():
            review_lengths.setdefault(rating, []).append(len(review.split()))
        if title:
            titles.setdefault(rating, []).append({"id": event["id"], "title": title[:100]})
    selected_ids = {event["id"] for event in events[-18:]}
    for group in buckets.values():
        selected_ids.update(group[index]["id"] for index in (0, len(group) // 2, len(group) - 1))
    selected = []
    for event in events:
        if event["id"] not in selected_ids:
            continue
        details = event["event"]
        product = details.get("product") or {}
        review = str(details.get("review") or "")
        selected.append({"id": event["id"], "ts": event.get("ts"),
            "domain": str(details.get("domain") or event.get("app") or "unknown")[:120],
            "rating": details.get("rating"), "review_title": str(details.get("review_title") or "")[:160],
            "product": {"title": str(product.get("title") or "")[:240],
                        "category": str(product.get("category") or "")[:160]},
            "review_excerpt": review[:480] + (" ... " + review[-130:] if len(review) > 650 else review[480:])})
    while len(json.dumps(selected, ensure_ascii=False)) > 34000:
        selected = selected[1::2]
    category_stats = [{"domain": domain, "category": category, "count": len(scores),
        "mean_rating": round(sum(scores) / len(scores), 2)}
        for (domain, category), scores in sorted(categories.items(), key=lambda item: (-len(item[1]), item[0]))
        if len(scores) >= 3][:24]
    ledger = [{"domain": domain, "rating": rating, "examples": examples[::max(1, len(examples) // 5)][:5]}
        for (domain, rating), examples in sorted(products.items(), key=lambda item: (-len(item[1]), item[0]))[:24]]
    styles = []
    for rating in sorted(set(review_lengths) | set(titles)):
        lengths = sorted(review_lengths.get(rating, []))
        examples = titles.get(rating, [])
        styles.append({"rating": rating, "count": len(lengths),
            "review_words_median": lengths[len(lengths) // 2] if lengths else None,
            "review_words_quartiles": [lengths[len(lengths) // 4], lengths[3 * len(lengths) // 4]] if lengths else [],
            "actual_titles": examples[::max(1, len(examples) // 8)][:8]})
    return json.dumps({"total_history_events": len(events), "selected_events": len(selected),
        "history_statistics": {"rating_count": len(ratings), "rating_counts": rating_counts,
            "rating_mean": sum(ratings) / len(ratings) if ratings else None,
            "title_count": len(title_lengths),
            "title_words_mean": sum(title_lengths) / len(title_lengths) if title_lengths else None,
            "title_word_bins": title_bins, "domain_counts": domains,
            "category_ratings": category_stats, "review_style_by_rating": styles},
        "rated_product_examples": ledger,
        "selection": "domain/rating first-middle-last plus recent18; statistics use all legal construction history; examples bounded; source ids retained",
        "events": selected}, ensure_ascii=False)