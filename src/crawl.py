import requests
import json
import time

def fetch_submissions(venue_id, offset=0, limit=1000):
    url = "https://api2.openreview.net/notes"
    params = {
        "content.venueid": venue_id,
        "details": "replyCount,invitation",
        "limit": limit,
        "offset": offset,
        "sort": "number:desc"
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()

def crawl_papers(venue_id, output_file):
    all_papers = []
    offset = 0
    limit = 1000
    
    print(f"Fetching papers from {venue_id}...")
    
    while True:
        data = fetch_submissions(venue_id, offset, limit)
        notes = data.get("notes", [])
        
        if not notes:
            break
            
        for note in notes:
            paper = {
                "id": note.get("id"),
                "number": note.get("number"),
                "title": note.get("content", {}).get("title", {}).get("value", ""),
                "authors": note.get("content", {}).get("authors", {}).get("value", []),
                "abstract": note.get("content", {}).get("abstract", {}).get("value", ""),
                "keywords": note.get("content", {}).get("keywords", {}).get("value", []),
                "primary_area": note.get("content", {}).get("primary_area", {}).get("value", ""),
                "forum_url": f"https://openreview.net/forum?id={note.get('id')}"
            }
            all_papers.append(paper)
        
        print(f"Fetched {len(notes)} papers (total: {len(all_papers)})")
        
        if len(notes) < limit:
            break
            
        offset += limit
        time.sleep(0.5)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)
    
    print(f"\nTotal: {len(all_papers)} papers")
    print(f"Saved to {output_file}")
    return all_papers

if __name__ == "__main__":
    crawl_papers(
        venue_id="ICLR.cc/2026/Conference/Submission",
        output_file="iclr2026_papers.json"
    )

