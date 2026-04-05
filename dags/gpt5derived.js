graph TD

%% ===== START =====
B0["B – Warm greeting and invite user to share what they want today"]

%% First user responses (including opt-out)
B0 --> U1a["U – Describes a specific concern or symptom"]
B0 --> U1b["U – Unsure what they want or how they feel"]
B0 --> U1c["U – Expresses intense distress or possible crisis"]
B0 --> Ustop["U – Wants to stop or pause the conversation"]

%% Bot responds to initial patterns
U1a --> B2a["B – Validate concern and ask about current emotions and intensity"]
U1b --> B2b["B – Normalize confusion and present simple menu of options"]
U1c --> B2c["B – Acknowledge distress and gently assess immediate safety"]
Ustop --> Bstop["B – Respect decision, offer brief reassurance, and share crisis options if needed"]

%% Branch from specific concern
B2a --> U3a1["U – Openly explores emotions in more detail"]
B2a --> U3a2["U – Answers very briefly or avoids sharing details"]
B2a --> U3a3["U – Says talking about it feels overwhelming"]
B2a --> U3a4["U – Describes flashback triggered by a current sound or cue"]
B2a --> U3a5["U – Describes distressing nightmare that feels real"]
B2a --> U3a6["U – Reports body memory such as tightness or pressure"]
B2a --> U3d1["U – Describes feeling unreal, far away, or unable to feel body"]

%% Branch from uncertainty about needs
B2b --> U3b1["U – Chooses grounding exercise"]
B2b --> U3b2["U – Chooses to talk things through"]
B2b --> U3b3["U – Asks bot to decide or stays unsure"]
B2b --> U3d1

%% Branch from crisis signal
B2c --> U3c1["U – Indicates immediate danger to self or others"]
B2c --> U3c2["U – Denies immediate danger but feels highly distressed"]

%% Crisis handling (terminal)
U3c1 --> B4crisis["B – Focus on safety and direct user to crisis or emergency services"]

%% Talk-focused path
U3a1 --> B4talk["B – Reflect main themes and invite feelings-focused exploration"]
U3b2 --> B4talk

B4talk --> U5t1["U – Shares detailed story and context"]
B4talk --> U5t2["U – Stays vague, changes topic, or avoids details"]
B4talk --> Uq1["U – Asks a direct question about feelings or coping"]

Uq1 --> Bq1["B – Answer question with validating, plain-language information"]
Bq1 --> U5t1

%% Guarded but not refusing
U3a2 --> B4gentle["B – Acknowledge guardedness and offer smaller sharing or grounding"]

B4gentle --> U5ge1["U – Agrees to share a little more"]
B4gentle --> U5ge2["U – Prefers grounding over talking"]

U5ge1 --> B4talk
U5ge2 --> B4groundIntro

%% Overwhelmed / choosing grounding / distressed but safe
U3a3 --> B4groundIntro["B – Suggest grounding to help with current overwhelm"]
U3b1 --> B4groundIntro
U3b3 --> B4suggest["B – Suggest brief grounding when user is unsure what they want"]
U3c2 --> B4groundIntro

B4suggest --> U5g1["U – Accepts brief grounding as a starting point"]
B4suggest --> U5t1

%% Flashback-focused grounding
U3a4 --> B7flashback["B – Use room-based grounding: notice current objects, differences from past, and time/place (‘That was then, this is now; I’m in the UK, it’s 2025’)"]
B7flashback --> U7g1["U – Feels a bit calmer or finds exercise somewhat helpful"]
B7flashback --> U7g2["U – Reports little change or increased discomfort"]

%% Nightmare recovery grounding
U3a5 --> B7nightmare["B – Use post-nightmare grounding: notice room, bed, pillow, and remind self it was a dream in the present (UK, 2025)"]
B7nightmare --> U7g1
B7nightmare --> U7g2

%% Tactile/body memory grounding
U3a6 --> B7tactile["B – Use body-based grounding: feet on floor, own hand on neck, slow breathing and safety statements"]
B7tactile --> U7g1
B7tactile --> U7g2

%% Dissociation protocol
U3d1 --> B7dissoc1["B – Acknowledge dissociation and invite small movements while focusing on bot’s voice"]
B7dissoc1 --> Ud1["U – Tries movements (e.g., crossing legs, squeezing fists) and focuses on voice as able"]
Ud1 --> B7dissoc2["B – Guide visual focus on nearby objects and colours (e.g., phone screen, shoes)"]
B7dissoc2 --> Ud2["U – Notices some return of senses or awareness"]
Ud2 --> B7dissoc3["B – Orient to name, room, year, and current safety (‘It’s 2025, I’m safe, I’m in the present’)"]
B7dissoc3 --> U7g1

%% Grounding introduction (generic)
B4groundIntro --> U5g1
B4groundIntro --> U5g2["U – Declines grounding; prefers talking or practical tips"]

%% Grounding execution (generic, with brief explanation)
U5g1 --> B6guide["B – Explain simple grounding and guide step-by-step exercise"]

B6guide --> U7g1
B6guide --> U7g2["U – Reports little change or increased discomfort"]

%% Alternative to grounding
U5g2 --> B6alt["B – Shift to practical, action-focused problem-solving"]

B6alt --> U5t1
B6alt --> U5t2

%% Formulating and relationship building in talk path
U5t1 --> B6formulate["B – Summarize themes and check if short-term goals feel right"]
U5t2 --> B6buildTrust["B – Check comfort level, validate hesitation, and ask smaller questions"]

%% Responses to formulation / trust building
B6formulate --> U9wrap1["U – Feels goals or focus are helpful"]
B6formulate --> U9wrap2["U – Unsure about plan or still ambivalent"]

B6buildTrust --> U9wrap1
B6buildTrust --> U9wrap2

%% Responses to grounding outcome (generic + specialised)
U7g1 --> B8reinforce["B – Reinforce progress and suggest next small steps"]
U7g2 --> B8adjust["B – Validate difficulty and adapt coping suggestions"]

B8reinforce --> U9wrap1
B8adjust --> U9wrap2

%% Closing paths (terminal)
U9wrap1 --> B9plan["B – Summarize session, confirm supports, and offer optional resources or next steps"]
U9wrap2 --> B9normalize["B – Normalize ongoing process and suggest one small experiment to try"]

%% ===== STYLES =====
classDef botNode fill:#e0f7fa,stroke:#00796b,stroke-width:1px;
classDef userNode fill:#fff3e0,stroke:#ef6c00,stroke-width:1px;

class B0,B2a,B2b,B2c,B4crisis,B4talk,B4gentle,B4groundIntro,B4suggest,B6guide,B6alt,B6formulate,B6buildTrust,B8reinforce,B8adjust,B9plan,B9normalize,Bstop,Bq1,B7flashback,B7nightmare,B7tactile,B7dissoc1,B7dissoc2,B7dissoc3 botNode;
class U1a,U1b,U1c,U3a1,U3a2,U3a3,U3a4,U3a5,U3a6,U3b1,U3b2,U3b3,U3c1,U3c2,U3d1,U5g1,U5g2,U5ge1,U5ge2,U5t1,U5t2,U7g1,U7g2,U9wrap1,U9wrap2,Ustop,Uq1,Ud1,Ud2 userNode;
