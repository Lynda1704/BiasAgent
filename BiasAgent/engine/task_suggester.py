from typing import Dict, List, Any


def normalize_text(text: str) -> str:
    return text.strip().lower()


def collect_task_candidates(tasks_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return tasks_data.get("tasks", [])


def suggest_task_from_metadata_and_labels(
    model_metadata: Dict[str, Any],
    tasks_data: Dict[str, Any],
    model_labels: List[str]
) -> Dict[str, Any]:

    model_name = normalize_text(model_metadata.get("model_id", ""))
    pipeline_tag = normalize_text(model_metadata.get("pipeline_tag", ""))
    tags = [normalize_text(t) for t in (model_metadata.get("tags") or [])]
    model_labels_norm = [normalize_text(x) for x in model_labels]

    candidates = collect_task_candidates(tasks_data)

    scored_candidates = []

    for task in candidates:
        task_name = normalize_text(task.get("name", ""))
        aliases = [normalize_text(a) for a in task.get("aliases", [])]

        score = 0
        reasons = []


        if task_name in model_name:
            score += 4
            reasons.append(f"task name '{task_name}' appears in model name")

        for alias in aliases:
            if alias in model_name:
                score += 3
                reasons.append(f"alias '{alias}' appears in model name")


        if task_name in tags:
            score += 4
            reasons.append(f"task name '{task_name}' appears in model tags")

        for alias in aliases:
            if alias in tags:
                score += 2
                reasons.append(f"alias '{alias}' appears in model tags")


        label_hits = 0
        for alias in aliases:
            for ml in model_labels_norm:
                if alias == ml:
                    label_hits += 1
                elif alias in ml or ml in alias:
                    label_hits += 1

        if label_hits > 0:
            score += label_hits
            reasons.append(f"{label_hits} label-alignment hints found from observed model labels")

        scored_candidates.append({
            "task_name": task.get("name", ""),
            "score": score,
            "reasons": reasons
        })

    scored_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)

    if not scored_candidates:
        return {
            "task_suggestion_possible": False,
            "task_name": None,
            "confidence": "low",
            "reason": "No task candidates available."
        }

    best = scored_candidates[0]
    second_score = scored_candidates[1]["score"] if len(scored_candidates) > 1 else -1

    if best["score"] <= 0:
        confidence = "low"
        possible = False
    else:
        possible = True
        margin = best["score"] - second_score
        if best["score"] >= 6 and margin >= 2:
            confidence = "high"
        elif best["score"] >= 3:
            confidence = "medium"
        else:
            confidence = "low"

    return {
        "task_suggestion_possible": possible,
        "task_name": best["task_name"],
        "confidence": confidence,
        "reason": "; ".join(best["reasons"]) if best["reasons"] else "Weak heuristic evidence only.",
        "ranked_candidates": scored_candidates[:5]
    }
