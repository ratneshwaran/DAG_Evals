graph TD
    %% =========================================
    %% 1.  ENTRY: AFTER A FLASHBACK OR NIGHTMARE
    %% =========================================
    B0([Bot: Invite present-moment grounding]):::botNode

    U1{{User: Describes flashback trigger}}:::userNode
    U2{{User: Describes nightmare}}:::userNode
    U3{{User: Reports body memory}}:::userNode
    U4{{User: Reports dissociation warning}}:::userNode

    B0 --> U1
    B0 --> U2
    B0 --> U3
    B0 --> U4

    %% ---------- bot paths ----------
    B1([Bot: Flashback room-look protocol]):::botNode
    B2([Bot: Nightmare wake-up protocol]):::botNode
    B3([Bot: Body-memory safety check]):::botNode
    B5([Bot: Dissociation re-activation]):::botNode

    U1 --> B1
    U2 --> B2
    U3 --> B3
    U4 --> B5

    %% =========================================
    %% 2.  SHARED GROUNDING LOOP
    %% =========================================
    U6{{User: Names present items}}:::userNode
    B1 --> U6
    B2 --> U6
    B3 --> U6

    B6([Bot: Anchor with 3-details & safety phrase]):::botNode
    U6 --> B6

    U7{{User: Repeats safety phrase}}:::userNode
    B6 --> U7

    %% =========================================
    %% 3.  DISSOCIATION BRANCH (parallel)
    %% =========================================
    U8{{User: Performs movements}}:::userNode
    B5 --> U8

    B7([Bot: Colour & name checks]):::botNode
    U8 --> B7

    U9{{User: Answers colour/name}}:::userNode
    B7 --> U9

    B8([Bot: Feet-press & present-date]):::botNode
    U9 --> B8

    U10{{User: Feels grounded}}:::userNode
    B6 --> U10
    B8 --> U10

    %% =========================================
    %% 4.  OPTIONAL NEXT STEPS
    %% =========================================
    U11{{User: Wants coping plan}}:::userNode
    U12{{User: Requests human therapist}}:::userNode
    U10 --> U11
    U10 --> U12

    B11([Bot: Offer take-home tips]):::botNode
    B12([Bot: Provide crisis/human hand-off]):::botNode
    U11 --> B11
    U12 --> B12

    U13{{User: Ready to finish}}:::userNode
    B11 --> U13
    B12 --> U13

    B14([Bot: Close session]):::botNode
    U13 --> B14
