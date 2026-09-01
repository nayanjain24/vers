# VERS v5.0 — Complete Sign Language Dictionary & Gesture Guide

This guide details all **30+ supported hand geometries, signs, and gestures** recognized by the VERS Multimodal AI Engine. The system processes 3D MediaPipe hand landmarks in real-time, matching against deterministic kinematic geometry, a 31-class neural temporal model, and soft pose centroids.

---

## 🚨 1. Emergency Signs (`EMERGENCY_SIGNS`)

These signs indicate imminent distress, safety hazards, physical injury, or police/fire/medical emergencies. Detecting these signs activates high-urgency telemetry, audible Siren TTS dispatch, and GPS emergency logging.

| Sign / Word | Severity | Hand Geometry & Physical Execution | ASL & ISL Variants | System Action |
| :--- | :--- | :--- | :--- | :--- |
| **HELP / SOS** | `Critical` | **Open hand with all 5 fingers spread wide** facing camera, or dominant fist resting on flat non-dominant palm with upward lift motion. | **ASL**: Open-5 hand moving urgently.<br>**ISL**: Clasped interlocking hands or rapid waving. | Immediate SOS dispatch + Siren voice announcement + emergency logs. |
| **MEDICAL** | `Critical` | **4 fingers extended grouped together** (Index, Middle, Ring, Pinky) with thumb folded tightly across the palm. | **ASL**: 4-finger 'M' shape near wrist/chest.<br>**ISL**: Index fingers crossed forming a '+' medical cross. | Medical alert dispatched to paramedics with vital telemetry. |
| **AMBULANCE** | `Critical` | **3 fingers extended** (Index, Middle, Ring) with thumb & pinky tucked, sweeping from side to side. | **ASL**: 3-finger extension with sweeping emergency siren light motion. | Urgent dispatch requesting immediate ambulance assistance. |
| **ACCIDENT** | `Critical` | **Tight solid fist**, all fingers curled inward tightly covering the thumb, held firmly in camera view. | **ASL**: Clenched 'S' fist or knuckles colliding.<br>**ISL**: Two closed fists moving toward each other. | Traffic/physical crash incident flagged with critical urgency. |
| **FIRE** | `High` | **Index finger pointing straight UP** alone, with fluttering/waving upward flicker motion mimicking flames. | **ASL**: Index finger up with oscillating wave.<br>**ISL**: Both hands with spread fingers flickering upwards. | Fire emergency alert generated; fire department notification prepared. |
| **POLICE** | `High` | **Index and Middle fingers extended in V-shape** (Peace / 2-finger salute) pointing upward. | **ASL**: V-shape held steady near chest or temple.<br>**ISL**: Index and middle fingers extended pointing forward. | Security threat flagged; police dispatch logged. |
| **DANGER** | `High` | **Closed fist with wrist rotated heavily** on the z-axis (facing away/down) near torso. | **ASL**: Fist crossing torso with downward snap.<br>**ISL**: Fist crossed over chest indicating hazard. | High threat flagged with urgent dispatch. |
| **PAIN** | `Medium` | **Partially clenched claw shape** (fingers curved like a claw) directed towards the chest or injury site. | **ASL**: Index fingers twisting toward each other.<br>**ISL**: Clawed palm arched tightly over body. | Medical pain flagged, prompts status check. |
| **FALL** | `High` | **Open hand with all 5 fingers pointing straight DOWN** vertically towards the ground. | **ASL**: 'V' fingers pointing downward (representing legs falling).<br>**ISL**: Flat palm brushing downward rapidly. | Fall detection incident triggered; inactivity timer started. |
| **STOP** | `Medium` | **Flat palm facing forward** with fingers held tightly together (rigid halt barrier). | **ASL**: Dominant hand chopping into flat non-dominant palm.<br>**ISL**: Flat palm facing front. | Emergency sequence paused / hold current protocol. |
| **SAFE** | `Low` | **Thumbs UP with tight fist** held steady in front of camera. | **ASL**: Thumbs up or crossing hands breaking apart.<br>**ISL**: Tight fist with thumb up held steady. | All-clear confirmed; downgrades threat level to safe. |

---

## 💬 2. Conversational & Needs Signs (`CONVERSATION_SIGNS`)

These signs allow mute, deaf, or speech-impaired individuals to communicate daily needs, polite requests, and conversational feedback without triggering siren alarms.

| Sign / Word | Category | Hand Geometry & Physical Execution | Meaning & Usage |
| :--- | :--- | :--- | :--- |
| **HELLO** | Conversational | **Open hand waving side to side** near temple/forehead. | Standard friendly greeting / start conversation. |
| **THANK_YOU** | Conversational | **Flat hand touching chin** and moving outward toward camera. | Expressing gratitude and politeness. |
| **PLEASE** | Conversational | **Flat palm held against the center of chest** moving in circular motion. | Polite request modifier (e.g., "Water Please"). |
| **YES** | Conversational | **Closed fist (ASL 'S') nodding up and down** mimicking a head nod. | Affirmative agreement / confirmation response. |
| **NO** | Conversational | **Index + Middle finger snapping against the thumb** repeatedly. | Negative denial / refusal response. |
| **WATER** | Daily Needs | **W-handshape (3 middle fingers extended, thumb holding pinky)** tapping chin twice. | Requesting drinking water or hydration. |
| **FOOD** | Daily Needs | **Flat-O handshape (all fingertips touching thumb)** tapping lips/mouth. | Requesting food, meal, or feeding assistance. |
| **WANT** | Daily Needs | **Both hands open clawed palms facing up**, pulling inward towards body. | Expressing a desire or immediate need. |
| **MORE** | Daily Needs | **Both hands in Flat-O shape, tapping fingertips together** repeatedly. | Requesting more food, water, or medicine. |
| **PHONE** | Daily Needs | **Y-handshape (Thumb and Pinky extended, middle 3 curled)** held to ear. | Requesting a phone call or mobile device. |
| **FRIEND** | Social | **Index fingers hooked into each other** and reversed. | Referring to a friend, ally, or companion. |
| **FAMILY** | Social | **F-handshapes (Index + Thumb touching in circle)** moving outward together. | Inquiring about or mentioning family members. |
| **NAME** | Social | **H-handshapes (Index + Middle extended together)** tapping across each other. | Asking for or stating an identity/name. |
| **GOOD** | Feedback | **Fingers of dominant hand touching chin, then moving down to flat palm**. | Positive status, approval, or feeling well. |
| **BAD** | Feedback | **Fingers on chin moving down while flipping palm downwards**. | Negative status, disapproval, or feeling unwell. |
| **SORRY** | Feedback | **Closed A-fist rubbing in a circular motion** over the chest/heart. | Sincere apology or expression of regret. |
| **UNDERSTAND** | Conversational | **Index finger flicking up like a lightbulb** near forehead. | Confirming comprehension of message. |
| **WHERE** | Conversational | **Index finger held upright wagging side-to-side**. | Location inquiry (e.g., "Where doctor?"). |
| **FINISHED** | Feedback | **Both open hands held palms-in, flipping outwards** away from body. | Task or meal finished, complete, done. |

---

## 🧠 3. How the Multimodal Engine Works

1. **Hardware & Direct Browser Webcam**: MediaPipe processes frames at 25–30 FPS, extracting 21 3D hand landmarks and 468 Face Mesh landmarks.
2. **Dual-Mode Filtering**:
   - **Mode: All Signs (Normal + Emergency)**: Detects both daily conversation signs and emergency gestures.
   - **Mode: Strict Emergency Only**: Whitelists critical crisis signals (`SOS`, `MEDICAL`, `FIRE`, etc.) to eliminate all false positives in high-risk zones.
3. **Compound Intent Reasoning**:
   - `["WATER", "PLEASE"]` ➔ Intent: **WATER_REQUEST [INFO]** (Polite hydration dispatch)
   - `["HELP", "ACCIDENT"]` ➔ Intent: **CRITICAL_ACCIDENT [CRITICAL]** (Instant sirens + emergency dispatch)
   - `["MEDICAL", "PAIN"]` ➔ Intent: **MEDICAL_ASSISTANCE [HIGH]** (Paramedic escalation)
   - `["SAFE"]` ➔ Intent: **ALL_CLEAR [LOW]** (Acknowledged status)
