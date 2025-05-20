

# Knowledge base and symbolic reasoning logic

KNOWLEDGE_BASE = {
    "threat_types_details": {
        "IED_urban": {
            "urgency": "high",
            "potential_impact": "severe",
            "indicators": ["suspicious_package", "chatter_increase"]
        },
        "armed_assault_gov": {
            "urgency": "critical",
            "potential_impact": "severe",
            "indicators": ["weapon_sighting", "reconnaissance_activity"]
        },
        "cyber_attack_infra": {
            "urgency": "high",
            "potential_impact": "high",
            "indicators": ["phishing_attempts", "network_anomaly"]
        },
        "propaganda_online": {
            "urgency": "medium",
            "potential_impact": "moderate",
            "indicators": ["fake_news_spike", "hate_speech_increase"]
        }
    },
    "countermeasures": {
        "cm_001": {
            "name": "Increased Patrols (Visible Presence)",
            "resources_needed": ["personnel_low"],
            "mitigates_threats": ["IED_urban", "armed_assault_gov"],
            "effectiveness_score": 0.6,
            "ethical_impact": "low",
            "legality_check": "standard_procedure"
        },
        "cm_002": {
            "name": "EOD Sweeps & Checkpoints",
            "resources_needed": ["personnel_medium", "eod_team", "equipment_basic"],
            "mitigates_threats": ["IED_urban"],
            "effectiveness_score": 0.85,
            "ethical_impact": "medium_low",
            "legality_check": "requires_justification"
        },
        "cm_003": {
            "name": "Enhanced Cybersecurity Monitoring",
            "resources_needed": ["cyber_team_skilled", "monitoring_tools_advanced"],
            "mitigates_threats": ["cyber_attack_infra"],
            "effectiveness_score": 0.75,
            "ethical_impact": "low",
            "legality_check": "privacy_compliant"
        },
        "cm_004": {
            "name": "Public Awareness Campaign (Disinformation)",
            "resources_needed": ["media_team", "communication_channels"],
            "mitigates_threats": ["propaganda_online", "IED_urban"],
            "effectiveness_score": 0.5,
            "ethical_impact": "very_low",
            "legality_check": "standard_procedure"
        },
        "cm_005": {
            "name": "Targeted Intelligence Operation",
            "resources_needed": ["intel_team_specialized", "surveillance_auth"],
            "mitigates_threats": ["armed_assault_gov", "IED_urban"],
            "effectiveness_score": 0.9,
            "ethical_impact": "high",
            "legality_check": "strict_oversight_warrant_required"
        },
        "cm_006": {
            "name": "De-escalation & Community Engagement",
            "resources_needed": ["community_liaisons", "negotiators"],
            "mitigates_threats": ["propaganda_online"],
            "effectiveness_score": 0.65,
            "ethical_impact": "positive",
            "legality_check": "standard_procedure"
        }
    }
}

class SymbolicCountermeasureSuggester:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def get_threat_details(self, threat_type_id):
        return self.kb.get("threat_types_details", {}).get(threat_type_id, {})

    def suggest_countermeasures(self, forecast):
        threat_type_id = forecast.get("threat_type_id")
        threat_details = self.get_threat_details(threat_type_id)
        suggestions = []

        for cm_id, cm_data in self.kb.get("countermeasures", {}).items():
            if threat_type_id in cm_data.get("mitigates_threats", []):
                base_score = cm_data.get("effectiveness_score", 0.5)
                prediction_score = forecast.get("prob", 0.5)
                score = base_score * (prediction_score + 0.5)

                suggestions.append({
                    "id": cm_id,
                    "name": cm_data.get("name"),
                    "score": round(score, 3),
                    "ethical_impact": cm_data.get("ethical_impact"),
                    "legality_check": cm_data.get("legality_check"),
                    "resources": cm_data.get("resources_needed", [])
                })

        return sorted(suggestions, key=lambda x: x["score"], reverse=True)
