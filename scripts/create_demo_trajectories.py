from __future__ import annotations

import argparse
from pathlib import Path

from deep_research_review_v2.trajectory import (
    ACTION_SPACE,
    AgentAction,
    ClaimTrajectory,
    EvidenceItem,
    TrajectoryRecorder,
    TrajectoryStep,
)


def build_demo_trajectories():
    return [
        ClaimTrajectory(
            trajectory_id="demo-001",
            claim="Urban tree canopy reduces summer surface temperature in dense cities.",
            hypothesis="Tree canopy is associated with lower surface temperatures during heat events.",
            label="supported_with_caveats",
            metadata={"source": "demo"},
            steps=[
                TrajectoryStep(
                    step_id=1,
                    observation="The claim is broad and needs both supporting and counter evidence.",
                    candidate_actions=ACTION_SPACE,
                    action=AgentAction(
                        action_type="search",
                        argument="urban tree canopy heat island reduction study meta analysis",
                        rationale="Start with broad literature search for direct evidence on temperature effects.",
                    ),
                    should_stop=False,
                    stop_reason="Need initial evidence.",
                ),
                TrajectoryStep(
                    step_id=2,
                    observation="Found studies showing greener blocks have lower land-surface temperature, but effects vary by city form.",
                    candidate_actions=ACTION_SPACE,
                    action=AgentAction(
                        action_type="quote_evidence",
                        argument="Record strongest support study and note caveat on variation.",
                        rationale="Capture direct support before looking for limitations.",
                        evidence=[
                            EvidenceItem(
                                doc_id="paper-a",
                                snippet="Neighborhoods with higher canopy cover showed lower summer land-surface temperature.",
                                stance="support",
                                score=0.92,
                            )
                        ],
                    ),
                    should_stop=False,
                    stop_reason="Need contradictory or limiting evidence.",
                ),
                TrajectoryStep(
                    step_id=3,
                    observation="Support evidence exists, but there is little on cases where canopy alone is insufficient.",
                    candidate_actions=ACTION_SPACE,
                    action=AgentAction(
                        action_type="search",
                        argument="urban canopy cooling limitations humidity wind morphology study",
                        rationale="Look for counter-evidence and failure cases.",
                    ),
                    should_stop=False,
                    stop_reason="Counter-evidence missing.",
                ),
                TrajectoryStep(
                    step_id=4,
                    observation="Found evidence that cooling gains depend on background density, wind, and irrigation conditions.",
                    candidate_actions=ACTION_SPACE,
                    action=AgentAction(
                        action_type="update_graph",
                        argument="Add limitation nodes for density, wind, irrigation.",
                        rationale="The evidence graph should now represent both support and constraints.",
                        evidence=[
                            EvidenceItem(
                                doc_id="paper-b",
                                snippet="Cooling benefit changed with street geometry and moisture conditions.",
                                stance="contradict",
                                score=0.74,
                            )
                        ],
                        graph_update={
                            "supports": ["canopy lowers temperature"],
                            "limits": ["depends on morphology", "depends on moisture"],
                        },
                    ),
                    should_stop=True,
                    stop_reason="Enough evidence collected to issue a qualified conclusion.",
                ),
            ],
        ),
        ClaimTrajectory(
            trajectory_id="demo-002",
            claim="Peer instruction reliably improves undergraduate physics learning outcomes.",
            hypothesis="Interactive peer-instruction methods outperform lecture-only baselines in physics courses.",
            label="mostly_supported",
            metadata={"source": "demo"},
            steps=[
                TrajectoryStep(
                    step_id=1,
                    observation="Need direct learning-outcome evidence and possible failure cases.",
                    candidate_actions=ACTION_SPACE,
                    action=AgentAction(
                        action_type="search",
                        argument="peer instruction physics learning outcomes randomized study",
                        rationale="Gather direct benchmark evidence first.",
                    ),
                    should_stop=False,
                    stop_reason="No evidence yet.",
                ),
                TrajectoryStep(
                    step_id=2,
                    observation="Multiple studies report gains on conceptual inventories under peer instruction.",
                    candidate_actions=ACTION_SPACE,
                    action=AgentAction(
                        action_type="quote_evidence",
                        argument="Capture concept inventory improvements from peer instruction papers.",
                        rationale="Record core supportive evidence.",
                        evidence=[
                            EvidenceItem(
                                doc_id="paper-c",
                                snippet="Students in peer-instruction sections achieved higher concept inventory gains.",
                                stance="support",
                                score=0.95,
                            )
                        ],
                    ),
                    should_stop=False,
                    stop_reason="Need boundary conditions.",
                ),
                TrajectoryStep(
                    step_id=3,
                    observation="Support is strong, but implementation quality may determine effect size.",
                    candidate_actions=ACTION_SPACE,
                    action=AgentAction(
                        action_type="ask_followup",
                        argument="What studies examine instructor fidelity and class-size effects?",
                        rationale="Boundary conditions likely control transferability.",
                    ),
                    should_stop=False,
                    stop_reason="Need implementation caveats.",
                ),
                TrajectoryStep(
                    step_id=4,
                    observation="Implementation fidelity appears to mediate outcomes; evidence is sufficient for a qualified stop.",
                    candidate_actions=ACTION_SPACE,
                    action=AgentAction(
                        action_type="stop",
                        argument="Stop after qualified conclusion.",
                        rationale="The record now includes both support and implementation caveats.",
                        evidence=[
                            EvidenceItem(
                                doc_id="paper-d",
                                snippet="Effects weakened when peer discussion was poorly structured.",
                                stance="contradict",
                                score=0.68,
                            )
                        ],
                    ),
                    should_stop=True,
                    stop_reason="Support and boundary conditions are both represented.",
                ),
            ],
        ),
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    recorder = TrajectoryRecorder(args.output)
    recorder.extend(build_demo_trajectories())
    print(f"Wrote demo trajectories to {args.output}")


if __name__ == "__main__":
    main()
