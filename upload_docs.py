import requests
import json

url = "https://api.vectara.io/v2/corpora/zwt_Nu4PPPxqYnAHz3CGl8Ek-xZZFmtoQqJfY9QPcg/documents"

payload = json.dumps({
  "type": "structured",
  "id": "esg_report_2024",
  "title": "2024 ESG Annual Report – EuroBank",
  "description": "Comprehensive report on EuroBank’s environmental, social, and governance initiatives for 2024, outlining sustainability goals and compliance with EU regulations.",
  "metadata": {
    "doc_type": "esg_report",
    "region": "EU",
    "industry": "banking",
    "year": 2024,
    "status": "published",
    "owner": "sustainability_team"
  },
  "sections": [
    {
      "id": 1,
      "title": "Environmental Initiatives",
      "text": "٦٠٦ طريق الملك فهد، العليا، برج الإبداع، الطابق ٢٠، ١٣١٣٢ الرياض\nBidaya.com.sa / ٨٠٠ ١١٨ ٢٢٢١\nتقرير مجلس إدارة شركة بداية للتمويل للعام المالي ٢٠٢٣م",
      "metadata": {"chunk_id": "Board of Directors Annual Report (Year End 31-12-2023)_0_3262a0ad5a_p001_t000001", "doc_id": "Board of Directors Annual Report (Year End 31-12-2023)_0_3262a0ad5a"},
      "section_type": "social",
      "priority": "medium"
    },
    {
      "id": 2,
      "title": "Social Responsibility",
      "text": "EuroBank expanded its community outreach programs, supporting over 500 local initiatives and increasing employee volunteer hours by 30%.",
      "metadata": None,
      "section_type": "social",
      "priority": "medium"
    }
  ]
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'x-api-key': 'zut_Nu4PPKZDETwSSOdLB4G2P66fxaMTRju84XzZvA'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)