"""
LLM_Training.py — ONE FILE, THREE NEURAL LEARNING STAGES + OPTIONAL ADAPTATION
============================================================================

SIGNAL FLOW
  1. NEXT WORD: prior detailed words -> causal attention -> next word
  2. IDEA: a short situation -> sequence encoder -> 1 compact idea vector
           -> WHO / ACTION / WHAT / WHERE / WHEN / CONDITION labels
  3. DETAIL: learned idea vector + previous words -> detailed next words
  4. OPTIONAL: frozen slow model + fast adapter vs full-weight updates

YOUR ORIGINAL DATA: all 52 aligned idea/detail examples embedded below.
EXPANSION: 52 originals + 50 first additions + 185 more distinct events
= 287 core idea/detail pairs, then reproducible synthetic variations
produce approximately 8,000 paired examples by default. Synthetic
variation is NOT the same as 8,000 independently authored stories.

TRAINING: each stage receives shuffled MINIBATCHES for multiple epochs.
VALIDATION: measures withheld combinations from the same templates.
CHECKPOINT: save learned weights and resume without starting over.

INSTALL (ONCE):  python3 -m pip install torch
RUN:             python3 LLM_Training.py
CONTINUAL MODE:  python3 LLM_Training.py --mode session --resume
REVIEW:          python3 LLM_Training.py --mode review
LEARN:           python3 LLM_Training.py --mode learn
WEIGHTS-ONLY:    python3 LLM_Training.py --mode evaluate --weights-only --question "Who was Jake?"
SELF-TEST:       python3 LLM_Training.py --mode self-test
FAST SMOKE TEST: python3 LLM_Training.py --examples 120 --epochs 1 --layers 1
MORE TRAINING:   python3 LLM_Training.py --epochs 5 --idea-epochs 10 --batch-size 64
RESUME TRAIN:    python3 LLM_Training.py --resume --epochs 5
INFERENCE ONLY:  python3 LLM_Training.py --resume --epochs 0 --idea-epochs 0 --idea "the fox spots the rabbit"
TRACE WORDS:     python3 LLM_Training.py --resume --epochs 0 --idea-epochs 0 --assess
ADAPTATION:      python3 LLM_Training.py --resume --epochs 0 --idea-epochs 0 --adapt
CUSTOM DATA:     python3 LLM_Training.py --ideas-file ideas.txt --details-file details.txt

HONEST LIMITS: A word-token toy neural network, NOT general intelligence.
287 independent short situations are recombined into thousands of
formulaic examples. Accurate labels or copied styles do not prove that it
understands unfamiliar language or can reason about arbitrary situations.
PERSONAL LEARNING: Sessions record evidence and train ONLY user-approved
question/answer targets. Both existing idea-reader and writer weights receive
gradients, with public-data replay and retention checks. This is supervised
memorization research, not autonomous semantic understanding. A manual Q/A
label must be supported by its underlying evidence. Temporal information is
logged but time/causality reasoning is not automatically learned. Corrections
trigger rebuilding from an untouched general checkpoint when available.
"""

# =====================================================================
# 0. EMBEDDED ORIGINAL TRAINING DATA — The user's supplied 52 aligned pairs
# =====================================================================

BASE_IDEAS = """
# =====================================================================
# IDEA / SITUATION TRAINING DATASET (52 SITUATIONS)
# =====================================================================
# Format per line:
#   situation text | who | action | what
#
# Categories:
#   1. Domestic Animals & Pets
#   2. Wild Animals & Nature
#   3. Professions & Human Activities
#   4. Science & Technology
#   5. Exploration & Adventure
# =====================================================================

# --- 1. Domestic Animals & Pets ---
the cat sees the ball | cat | sees | ball
the cat chases the ball | cat | chases | ball
the cat catches the ball | cat | catches | ball
the dog sees the cat | dog | sees | cat
the dog chases the cat | dog | chases | cat
the dog follows the cat | dog | follows | cat
the ball rolls away | ball | rolls | none
the cat runs after the ball | cat | runs | ball
the dog runs after the cat | dog | runs | cat
the cat sleeps by the window | cat | sleeps | window
the dog sleeps by the door | dog | sleeps | door
the ball stops by the door | ball | stops | door
the cat follows the dog | cat | follows | dog
the dog catches the ball | dog | catches | ball
the dog sees the ball | dog | sees | ball
the cat sees the dog | cat | sees | dog

# --- 2. Wild Animals & Nature ---
the bird flies over the river | bird | flies | river
the bird spots the fish | bird | spots | fish
the bird catches the fish | bird | catches | fish
the fox hunts near the woods | fox | hunts | woods
the fox spots the rabbit | fox | spots | rabbit
the rabbit hides in the grass | rabbit | hides | grass
the wolf roams across the mountain | wolf | roams | mountain
the bear sleeps in the cave | bear | sleeps | cave
the deer drinks from the stream | deer | drinks | stream
the owl watches from the branch | owl | watches | branch

# --- 3. Professions & Human Activities ---
the chef cooks the dinner | chef | cooks | dinner
the chef slices the bread | chef | slices | bread
the doctor examines the patient | doctor | examines | patient
the doctor heals the sick | doctor | heals | sick
the artist paints the portrait | artist | paints | portrait
the artist sketches the landscape | artist | sketches | landscape
the detective inspects the clue | detective | inspects | clue
the detective solves the mystery | detective | solves | mystery
the gardener plants the flower | gardener | plants | flower
the gardener waters the garden | gardener | waters | garden
the student studies the lesson | student | studies | lesson
the student writes the essay | student | writes | essay

# --- 4. Science & Technology ---
the robot cleans the laboratory | robot | cleans | laboratory
the robot scans the sample | robot | scans | sample
the scientist tests the formula | scientist | tests | formula
the scientist analyzes the data | scientist | analyzes | data
the astronomer observes the planet | astronomer | observes | planet
the satellite orbits the earth | satellite | orbits | earth
the engineer repairs the engine | engineer | repairs | engine
the computer solves the problem | computer | solves | problem

# --- 5. Exploration & Adventure ---
the knight protects the castle | knight | protects | castle
the knight guards the gate | knight | guards | gate
the explorer enters the cave | explorer | enters | cave
the explorer finds the treasure | explorer | finds | treasure
the traveler climbs the hill | traveler | climbs | hill
the traveler watches the sunset | traveler | watches | sunset
"""

BASE_DETAILS = """
# =====================================================================
# DETAILED SENTENCE TRAINING DATASET (52 DETAILED SENTENCES)
# =====================================================================
# Each line pairs with the corresponding line in ideas.txt.
# The language model learns to generate these rich, detailed descriptions
# guided by the compressed idea vectors from ideas.txt.
# =====================================================================

# --- 1. Domestic Animals & Pets ---
the cat sees the ball and watches it closely .
the cat runs quickly after the rolling ball .
the cat catches the ball and holds it still .
the dog notices the cat near the door .
the dog runs quickly after the cat .
the dog walks slowly behind the cat .
the ball moves across the floor and rolls away .
the cat runs after the ball across the floor .
the dog runs after the cat across the floor .
the cat rests quietly next to the window .
the dog rests quietly near the door .
the ball stops moving beside the door .
the cat walks slowly behind the dog .
the dog catches the ball and holds it still .
the dog sees the ball and watches it closely .
the cat notices the dog near the window .

# --- 2. Wild Animals & Nature ---
the bird spreads wide wings and glides over the clear river .
the bird circles high above and spots the swimming silver fish .
the bird dives sharp into water and catches the slippery fish .
the fox treads soft on leaves and hunts near the dark woods .
the fox peers through tall grass and spots the small rabbit .
the rabbit stays still and hides deep inside the dry grass .
the wolf prowls alone and roams across the snowy rocky mountain .
the heavy brown bear rests deep inside the dark warm cave .
the graceful wild deer pauses and drinks fresh cold stream water .
the silent grey owl perches high and watches from the oak branch .

# --- 3. Professions & Human Activities ---
the skilled chef lights the stove and cooks a savory hot dinner .
the chef holds a sharp knife and slices the crusty warm bread .
the kind doctor checks vital signs and examines the sick patient .
the doctor provides medicine and gently heals the suffering sick child .
the creative artist blends rich colors and paints a detailed portrait .
the artist uses charcoal and quickly sketches the sunny green landscape .
the clever detective uses a magnifying glass and inspects the tiny clue .
the detective pieces evidence together and solves the difficult mystery .
the gardener turns fertile soil and plants a red fragrant flower .
the gardener holds a metal hose and waters the blooming garden .
the curious young student reads notes and studies the hard math lesson .
the student drafts paragraphs with ink and writes a thoughtful essay .

# --- 4. Science & Technology ---
the shiny metal robot moves quietly and cleans the modern laboratory .
the robot aims a blue sensor beam and scans the crystal sample .
the scientist mixes chemical beakers and tests the bubbling new formula .
the researcher views computer graphs and analyzes the experimental data .
the astronomer adjusts the mirror and observes the distant red planet .
the silver satellite circles through outer space and orbits planet earth .
the engineer tightens loose bolts and repairs the loud humming engine .
the fast silicon computer processes calculations and solves the complex problem .

# --- 5. Exploration & Adventure ---
the brave knight raises an iron shield and protects the tall castle .
the armored knight stands watchful guard before the heavy wooden gate .
the bold explorer carries a burning torch and enters the unknown cave .
the lucky explorer opens a golden chest and finds the hidden treasure .
the weary traveler reaches the top and climbs the steep grassy hill .
the peaceful traveler sits quietly and watches the glowing evening sunset .
"""

# =====================================================================
# 1. DATA — Original examples, synthetic situations, and minibatches
# =====================================================================

import argparse
import copy
import itertools
import math
import random
import shutil
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
FIELDS = ("WHO", "ACTION", "WHAT", "WHERE", "WHEN", "CONDITION")

# Default global values for configuration & backwards compatibility
DEFAULT_IDEAS_FILE = "ideas.txt"
DEFAULT_DETAILS_FILE = "details.txt"
DEFAULT_LAYERS = 2
VECTOR_SIZE = 64
CONTEXT = 36
LEARNING_RATE = 0.0015

# Your original paired datasets are embedded above as BASE_IDEAS / BASE_DETAILS.
# These additional authored situations expand the kinds of events available.
# Format: short situation | who | action | what || descriptive target
ADDITIONAL_EVENTS = {
    "pets": """
the puppy carries the toy | puppy | carries | toy || the cheerful puppy proudly carries a favorite colorful toy .
the kitten watches the bird | kitten | watches | bird || the curious little kitten studies the bird with bright alert eyes .
the horse pulls the cart | horse | pulls | cart || the strong horse steadily pulls the wooden cart forward .
the rabbit eats the carrot | rabbit | eats | carrot || the small rabbit happily nibbles the crunchy orange carrot .
the dog guards the house | dog | guards | house || the watchful dog stands alert and protects the quiet house .
the cat drinks the milk | cat | drinks | milk || the thirsty cat gently laps up the fresh milk .
the puppy follows the child | puppy | follows | child || the excited puppy stays close behind the laughing child .
the mouse hides under the table | mouse | hides | table || the tiny mouse slips safely beneath the wooden table .
the dog finds the toy | dog | finds | toy || the playful dog discovers a forgotten toy under the chair .
the kitten climbs the chair | kitten | climbs | chair || the energetic kitten carefully scrambles up the tall chair .
""",
    "nature": """
the eagle catches the fish | eagle | catches | fish || the powerful eagle swoops low and seizes the silver fish .
the turtle crosses the path | turtle | crosses | path || the patient turtle slowly travels across the narrow path .
the squirrel gathers the nuts | squirrel | gathers | nuts || the busy squirrel collects small nuts for the winter .
the dolphin follows the boat | dolphin | follows | boat || the playful dolphin swims beside the moving boat .
the deer follows the river | deer | follows | river || the cautious deer walks beside the twisting river .
the bear catches the fish | bear | catches | fish || the hungry bear reaches into the river and catches a fish .
the wolf follows the tracks | wolf | follows | tracks || the alert wolf carefully follows fresh tracks through the snow .
the bee visits the flower | bee | visits | flower || the busy bee lands gently on the bright yellow flower .
the owl finds the mouse | owl | finds | mouse || the silent owl spots a small mouse beneath the trees .
the fox crosses the field | fox | crosses | field || the clever fox moves quietly across the wide open field .
""",
    "people": """
the nurse comforts the patient | nurse | comforts | patient || the compassionate nurse gently reassures the worried patient .
the teacher explains the lesson | teacher | explains | lesson || the patient teacher clearly explains a difficult new lesson .
the musician plays the piano | musician | plays | piano || the talented musician performs a gentle melody on the piano .
the baker makes the bread | baker | makes | bread || the careful baker kneads fresh dough and bakes warm bread .
the firefighter rescues the child | firefighter | rescues | child || the brave firefighter safely carries the frightened child away .
the farmer waters the crops | farmer | waters | crops || the hardworking farmer gives the growing crops fresh water .
the pilot lands the plane | pilot | lands | plane || the experienced pilot carefully brings the aircraft to the runway .
the mechanic fixes the bicycle | mechanic | fixes | bicycle || the skilled mechanic tightens the bicycle chain and repairs the wheel .
the writer edits the story | writer | edits | story || the thoughtful writer revises the story and improves each paragraph .
the athlete throws the ball | athlete | throws | ball || the focused athlete throws the ball toward the distant target .
""",
    "technology": """
the robot sorts the tools | robot | sorts | tools || the precise robot organizes metal tools into separate trays .
the drone maps the field | drone | maps | field || the flying drone records aerial images of the wide field .
the scientist measures the temperature | scientist | measures | temperature || the careful scientist records the temperature with a digital sensor .
the programmer tests the code | programmer | tests | code || the diligent programmer runs checks and investigates unexpected software errors .
the computer predicts the weather | computer | predicts | weather || the fast computer examines atmospheric data and estimates tomorrow weather .
the engineer designs the bridge | engineer | designs | bridge || the experienced engineer draws structural plans for the strong bridge .
the robot opens the door | robot | opens | door || the helpful robot detects the handle and opens the heavy door .
the researcher compares the results | researcher | compares | results || the curious researcher studies two results and identifies important differences .
the sensor detects the movement | sensor | detects | movement || the sensitive electronic sensor registers sudden movement nearby .
the astronaut repairs the antenna | astronaut | repairs | antenna || the trained astronaut carefully repairs the damaged communication antenna .
""",
    "adventure": """
the sailor navigates the ship | sailor | navigates | ship || the experienced sailor steers the ship through rough blue waves .
the explorer crosses the canyon | explorer | crosses | canyon || the daring explorer carefully crosses the deep rocky canyon .
the knight finds the map | knight | finds | map || the armored knight discovers an ancient map inside a dusty box .
the traveler follows the trail | traveler | follows | trail || the tired traveler follows the winding trail toward a distant village .
the captain protects the crew | captain | protects | crew || the determined captain keeps the crew safe during the storm .
the scout observes the valley | scout | observes | valley || the quiet scout studies the wide valley from the high ridge .
the traveler enters the village | traveler | enters | village || the hopeful traveler reaches the village and walks through its gate .
the explorer studies the map | explorer | studies | map || the focused explorer studies a detailed map before choosing a route .
the sailor sees the island | sailor | sees | island || the watchful sailor spots a small green island in the distance .
the knight climbs the tower | knight | climbs | tower || the brave knight climbs the winding stairs of the old stone tower .
""",
}

# =====================================================================
# 1A. EXPANDED REAL EVENTS — 185 more distinct idea/detail pairs
# =====================================================================

# These are additional distinct events, not thousands of rewrites of one idea.
# Each line: short situation | WHO | ACTION | WHAT || detailed sentence.
EXPANDED_EVENTS = {
    "pets": """
the dog brings the leash | dog | brings | leash || the excited dog carries its leash to the door and waits for a walk .
the cat opens the cupboard | cat | opens | cupboard || the curious cat pushes the loose cupboard door with one paw .
the puppy learns to sit | puppy | learns | sit || the young puppy watches its owner and settles into a seated position .
the dog shares the blanket | dog | shares | blanket || the gentle dog moves over and makes room for the kitten on the blanket .
the kitten knocks over the cup | kitten | knocks | cup || the playful kitten touches the cup and sends it sliding off the table .
the rabbit digs a burrow | rabbit | digs | burrow || the rabbit scrapes loose earth away and begins a sheltered burrow .
the parrot copies the greeting | parrot | copies | greeting || the bright parrot repeats a familiar greeting after hearing its owner .
the horse waits at the fence | horse | waits | fence || the calm horse stands beside the fence while the farmer approaches .
the puppy hides the toy | puppy | hides | toy || the mischievous puppy pushes its toy beneath the couch for later .
the cat cleans its paws | cat | cleans | paws || the sleepy cat carefully licks dirt from its front paws .
the dog alerts the owner | dog | alerts | owner || the attentive dog barks near the back door when it hears a noise .
the puppy spills the water | puppy | spills | water || the puppy bumps the bowl and splashes water across the kitchen floor .
the turtle reaches the pond | turtle | reaches | pond || the slow turtle crosses the grass and slips into the shallow pond .
the cat jumps onto the shelf | cat | jumps | shelf || the agile cat leaps from the chair to the high wooden shelf .
the dog searches for the bone | dog | searches | bone || the determined dog sniffs the yard until it uncovers a buried bone .
the hamster stores the seeds | hamster | stores | seeds || the tiny hamster gathers seeds and carries them into its nest .
the kitten follows the shadow | kitten | follows | shadow || the fascinated kitten trots after a moving shadow on the floor .
the dog learns a signal | dog | learns | signal || the dog begins to respond when its owner raises an open hand .
the horse drinks from the trough | horse | drinks | trough || the thirsty horse lowers its head and drinks fresh water from the trough .
the cat watches the rain | cat | watches | rain || the quiet cat sits by the window and follows raindrops with its eyes .
the puppy loses the ball | puppy | loses | ball || the puppy drops the ball into tall grass and searches for it .
the dog carries the newspaper | dog | carries | newspaper || the well trained dog gently carries the folded newspaper across the porch .
the rabbit escapes the pen | rabbit | escapes | pen || the quick rabbit squeezes through a small opening in its wooden pen .
the bird builds a nest | bird | builds | nest || the little bird gathers twigs and arranges them into a sturdy nest .
the kitten discovers the mirror | kitten | discovers | mirror || the kitten approaches the mirror and pauses at its own reflection .
the dog waits for dinner | dog | waits | dinner || the hungry dog sits patiently beside its bowl before dinner .
the cat rescues the kitten | cat | rescues | kitten || the mother cat carries the stranded kitten back to a safe corner .
the puppy follows the scent | puppy | follows | scent || the eager puppy sniffs the ground and follows a fresh scent around the yard .
the horse carries the rider | horse | carries | rider || the steady horse walks along the trail with a rider on its back .
the dog avoids the puddle | dog | avoids | puddle || the careful dog steps around a muddy puddle beside the road .
the parrot chooses the perch | parrot | chooses | perch || the parrot leaves the open cage and lands on a nearby wooden perch .
the hamster escapes the cage | hamster | escapes | cage || the restless hamster squeezes through the open cage door at night .
the cat retrieves the ribbon | cat | retrieves | ribbon || the cat catches the loose ribbon and pulls it away from the chair .
the puppy greets the visitor | puppy | greets | visitor || the friendly puppy runs toward the visitor and wags its tail .
the dog protects the puppy | dog | protects | puppy || the older dog stands between the puppy and an approaching stranger .
""",
    "nature": """
the salmon swims upstream | salmon | swims | upstream || the determined salmon pushes against the river current toward its spawning ground .
the fox shelters its cubs | fox | shelters | cubs || the mother fox leads her young cubs into a dry hidden den .
the owl hears the mouse | owl | hears | mouse || the silent owl turns its head after hearing a mouse in the grass .
the wolf leads the pack | wolf | leads | pack || the experienced wolf moves ahead while the pack follows through the snow .
the bee carries the pollen | bee | carries | pollen || the small bee collects pollen from blossoms and brings it to the hive .
the ant builds a tunnel | ant | builds | tunnel || the worker ant moves grains of soil to widen an underground tunnel .
the spider repairs the web | spider | repairs | web || the patient spider replaces a broken strand in its delicate web .
the beaver blocks the stream | beaver | blocks | stream || the busy beaver packs branches across the stream to slow the water .
the otter opens the shell | otter | opens | shell || the clever otter strikes the hard shell against a smooth stone .
the crow hides the food | crow | hides | food || the watchful crow tucks a piece of food beneath fallen leaves .
the eagle circles the valley | eagle | circles | valley || the eagle rides a rising current of air above the broad valley .
the whale surfaces for air | whale | surfaces | air || the enormous whale rises through the waves and exhales before breathing .
the dolphin guides the calf | dolphin | guides | calf || the adult dolphin swims beside its calf and steers it toward calmer water .
the bear searches for berries | bear | searches | berries || the hungry bear explores the hillside and eats ripe wild berries .
the deer senses danger | deer | senses | danger || the alert deer freezes after hearing a sudden crack among the trees .
the rabbit avoids the fox | rabbit | avoids | fox || the frightened rabbit darts beneath thick bushes before the fox approaches .
the squirrel buries the acorn | squirrel | buries | acorn || the busy squirrel scratches a shallow hole and covers the acorn with soil .
the frog catches the insect | frog | catches | insect || the frog flicks its tongue toward an insect passing above the pond .
the dragonfly lands on the reed | dragonfly | lands | reed || the delicate dragonfly settles lightly on a reed beside the water .
the snake sheds its skin | snake | sheds | skin || the growing snake rubs against rough bark and slips out of its old skin .
the butterfly leaves the cocoon | butterfly | leaves | cocoon || the emerging butterfly opens its wings and rests before its first flight .
the hawk protects the nest | hawk | protects | nest || the watchful hawk circles above the nest when another bird approaches .
the wolf crosses the ice | wolf | crosses | ice || the cautious wolf tests the frozen surface before crossing the narrow stream .
the fox follows the footprints | fox | follows | footprints || the hunting fox tracks fresh footprints through the soft snow .
the rain fills the pond | rain | fills | pond || steady rainfall raises the water level in the shallow woodland pond .
the wind bends the tree | wind | bends | tree || a strong gust bends the young tree without breaking its flexible trunk .
the roots absorb the water | roots | absorb | water || the spreading roots draw moisture from the soil beneath the plant .
the seed sprouts after rain | seed | sprouts | rain || the seed absorbs water and sends a small green shoot above the ground .
the river erodes the bank | river | erodes | bank || fast moving water gradually carries loose soil away from the river bank .
the cloud covers the mountain | cloud | covers | mountain || a low cloud drifts across the ridge and hides the mountain peak .
the owl guards its nest | owl | guards | nest || the parent owl remains close to the nest while its chicks rest .
the salmon escapes the bear | salmon | escapes | bear || the salmon twists through the current and slips past the waiting bear .
the deer leads its fawn | deer | leads | fawn || the mother deer guides its young fawn across a quiet meadow .
the ants transport a leaf | ants | transport | leaf || several ants work together to carry a leaf along the forest floor .
the tide reveals the rocks | tide | reveals | rocks || the falling tide exposes dark rocks that were hidden beneath the waves .
""",
    "people": """
the mother reads the story | mother | reads | story || the mother turns the pages and reads a bedtime story to her child .
the child solves the puzzle | child | solves | puzzle || the child studies the shapes and fits the final piece into the puzzle .
the student asks a question | student | asks | question || the curious student raises a hand to ask about a confusing example .
the teacher corrects the mistake | teacher | corrects | mistake || the teacher explains the mistaken step and demonstrates a better method .
the nurse measures the pulse | nurse | measures | pulse || the nurse gently checks the patient pulse and records the result .
the doctor reviews the scan | doctor | reviews | scan || the physician examines the scan and explains the findings to the patient .
the patient describes the pain | patient | describes | pain || the patient points to the sore area and describes when the pain began .
the therapist listens to the client | therapist | listens | client || the therapist listens carefully as the client explains a difficult week .
the counselor teaches a skill | counselor | teaches | skill || the counselor demonstrates a grounding skill and practices it with the client .
the chef tastes the soup | chef | tastes | soup || the chef samples the soup and adds herbs to improve the flavor .
the baker measures the flour | baker | measures | flour || the baker weighs flour before combining the ingredients for fresh bread .
the driver avoids the obstacle | driver | avoids | obstacle || the alert driver slows the car and steers around debris on the road .
the mechanic diagnoses the noise | mechanic | diagnoses | noise || the mechanic listens to the engine and traces the unusual noise to a loose belt .
the farmer repairs the fence | farmer | repairs | fence || the farmer replaces a broken fence post before the animals escape .
the gardener removes the weeds | gardener | removes | weeds || the gardener pulls weeds from the soil so the seedlings receive more light .
the artist chooses the colors | artist | chooses | colors || the artist compares paint samples before selecting colors for the mural .
the musician practices the melody | musician | practices | melody || the musician repeats a difficult melody until each note sounds clear .
the dancer learns the routine | dancer | learns | routine || the dancer practices each step while following the instructor rhythm .
the actor remembers the line | actor | remembers | line || the actor pauses briefly and then delivers the forgotten line on stage .
the athlete recovers the ball | athlete | recovers | ball || the athlete races across the court and catches the loose ball .
the coach changes the strategy | coach | changes | strategy || the coach studies the defense and suggests a different play for the team .
the firefighter checks the alarm | firefighter | checks | alarm || the firefighter inspects the alarm system and confirms that its sensors respond .
the pilot avoids the storm | pilot | avoids | storm || the pilot changes course to keep the aircraft away from dangerous weather .
the captain contacts the harbor | captain | contacts | harbor || the ship captain radios the harbor before approaching the busy dock .
the librarian organizes the books | librarian | organizes | books || the librarian arranges returned books on the shelves in alphabetical order .
the reporter verifies the claim | reporter | verifies | claim || the reporter checks multiple sources before publishing the disputed claim .
the detective questions the witness | detective | questions | witness || the detective asks careful questions to clarify what the witness observed .
the judge considers the evidence | judge | considers | evidence || the judge listens to both sides and reviews the evidence before ruling .
the architect redesigns the room | architect | redesigns | room || the architect changes the floor plan to make the room more accessible .
the carpenter builds the table | carpenter | builds | table || the carpenter cuts wooden boards and fastens them into a sturdy table .
the plumber finds the leak | plumber | finds | leak || the plumber inspects the pipe and finds a small leak behind the wall .
the electrician replaces the wire | electrician | replaces | wire || the electrician turns off the power before replacing the damaged wire .
the customer returns the package | customer | returns | package || the customer brings the unopened package back to the store for a refund .
the cashier counts the change | cashier | counts | change || the cashier carefully counts coins and hands the correct change to the customer .
the neighbor helps the family | neighbor | helps | family || the neighbor carries groceries to the family after hearing they need assistance .
the friend remembers the promise | friend | remembers | promise || the friend changes plans to keep an important promise made last week .
the child loses the key | child | loses | key || the child searches every pocket after realizing the door key is missing .
the parent finds the key | parent | finds | key || the parent notices the lost key beneath the couch and returns it to the child .
the student revises the essay | student | revises | essay || the student improves the argument and corrects unclear sentences in the essay .
the teacher compares the answers | teacher | compares | answers || the teacher reviews two answers and explains why they lead to different results .
""",
    "technology": """
the robot recognizes the object | robot | recognizes | object || the robot examines the camera image and identifies the object on the table .
the robot drops the tool | robot | drops | tool || the robot loses its grip and releases the tool onto the floor .
the robot retries the grasp | robot | retries | grasp || the robot adjusts its grip and tries to pick up the tool again .
the camera detects the motion | camera | detects | motion || the camera compares consecutive frames and detects movement near the doorway .
the sensor reports a failure | sensor | reports | failure || the sensor notices an unexpected reading and sends a failure signal .
the controller adjusts the speed | controller | adjusts | speed || the controller measures the motor response and reduces its command speed .
the computer saves the document | computer | saves | document || the computer writes the edited document to storage before closing the application .
the program catches the error | program | catches | error || the program identifies an invalid input and prevents the operation from continuing .
the programmer fixes the bug | programmer | fixes | bug || the programmer reproduces the failure and corrects the faulty calculation .
the developer reviews the change | developer | reviews | change || the developer reads the proposed code change and checks for unintended effects .
the server rejects the request | server | rejects | request || the server denies the request because its authentication token is invalid .
the client retries the request | client | retries | request || the client waits briefly and sends the request again after a timeout .
the database retrieves the record | database | retrieves | record || the database uses an index to locate the requested record quickly .
the database restores the backup | database | restores | backup || the database loads an earlier backup after detecting damaged records .
the network loses the connection | network | loses | connection || the network connection drops when the wireless signal becomes weak .
the router chooses the path | router | chooses | path || the router consults its routing table and forwards the packet along a route .
the model predicts the token | model | predicts | token || the model produces probabilities for the next token after reading the current context .
the model updates the weights | model | updates | weights || the model adjusts trainable weights after comparing predictions with targets .
the attention layer selects the context | attention | selects | context || the attention layer weighs earlier tokens and combines their information .
the encoder compresses the situation | encoder | compresses | situation || the encoder converts the sentence into a compact vector that summarizes its contents .
the decoder expands the idea | decoder | expands | idea || the decoder uses an idea representation to generate a detailed sentence one token at a time .
the learner detects the mismatch | learner | detects | mismatch || the learner compares its prediction with the observation and measures the difference .
the planner evaluates the options | planner | evaluates | options || the planner estimates possible outcomes before selecting an action .
the agent tests the hypothesis | agent | tests | hypothesis || the agent runs a controlled experiment to see whether its hypothesis predicts the result .
the agent revises the plan | agent | revises | plan || the agent changes its plan after receiving evidence that the first approach failed .
the agent stores the memory | agent | stores | memory || the agent saves a summary of the experience for use in a later task .
the agent retrieves the memory | agent | retrieves | memory || the agent searches stored experiences for one that matches the current situation .
the robot navigates the hallway | robot | navigates | hallway || the robot uses its sensor readings to move safely through the hallway .
the drone avoids the tree | drone | avoids | tree || the drone changes direction when its camera detects a tree in its flight path .
the satellite detects the storm | satellite | detects | storm || the satellite records cloud patterns and identifies a developing storm .
the engineer tests the battery | engineer | tests | battery || the engineer measures battery voltage under load to evaluate its performance .
the researcher measures the error | researcher | measures | error || the researcher compares model predictions with known outcomes and calculates the error .
the scientist repeats the experiment | scientist | repeats | experiment || the scientist performs the experiment again to see whether the results remain consistent .
the technician calibrates the sensor | technician | calibrates | sensor || the technician adjusts the sensor using a known reference measurement .
the machine overheats the motor | machine | overheats | motor || the machine runs too long under heavy load and the motor temperature rises .
the cooling system lowers the temperature | cooling | lowers | temperature || the cooling system circulates air across the hot components to reduce their temperature .
the engineer replaces the bearing | engineer | replaces | bearing || the engineer removes the worn bearing and installs a smooth replacement .
the scientist compares two models | scientist | compares | models || the scientist evaluates two models on the same held out examples .
the validator finds the duplicate | validator | finds | duplicate || the validator detects a repeated record and marks it for review .
the scheduler distributes the tasks | scheduler | distributes | tasks || the scheduler assigns independent tasks to available workers in a queue .
""",
    "adventure": """
the explorer finds the compass | explorer | finds | compass || the explorer discovers a small brass compass beneath a pile of old maps .
the traveler loses the compass | traveler | loses | compass || the traveler searches the campsite after realizing the compass is missing .
the guide identifies the trail | guide | identifies | trail || the experienced guide studies the terrain and chooses a marked mountain trail .
the traveler crosses the bridge | traveler | crosses | bridge || the traveler carefully steps across the narrow bridge above the ravine .
the explorer hears the waterfall | explorer | hears | waterfall || the explorer follows the distant rushing sound toward a hidden waterfall .
the crew repairs the sail | crew | repairs | sail || the crew sews the torn sail so the ship can continue its journey .
the captain changes the course | captain | changes | course || the captain steers away from the reefs after checking the sea chart .
the sailor rescues the passenger | sailor | rescues | passenger || the sailor throws a rope to the struggling passenger and pulls them aboard .
the knight protects the messenger | knight | protects | messenger || the knight escorts the messenger safely through a dangerous forest passage .
the messenger delivers the letter | messenger | delivers | letter || the messenger travels through the rain and hands the sealed letter to the king .
the explorer opens the gate | explorer | opens | gate || the explorer pushes the rusted gate and reveals a hidden garden .
the traveler discovers the cave | traveler | discovers | cave || the traveler notices an opening behind the waterfall and enters the cave .
the guide warns the group | guide | warns | group || the guide sees unstable rocks and warns the group to keep away .
the scout maps the canyon | scout | maps | canyon || the scout sketches the canyon walls and notes a safe crossing point .
the explorer avoids the trap | explorer | avoids | trap || the explorer notices a loose stone and carefully steps around the hidden trap .
the climber secures the rope | climber | secures | rope || the climber ties the rope to a solid anchor before descending .
the explorer reads the inscription | explorer | reads | inscription || the explorer brushes dust from the stone and reads an ancient inscription .
the scholar translates the inscription | scholar | translates | inscription || the scholar compares familiar symbols and translates part of the inscription .
the traveler shares the water | traveler | shares | water || the traveler offers the last water to a companion during the long journey .
the crew discovers the island | crew | discovers | island || the ship crew spots an uncharted island rising beyond the morning mist .
the sailor follows the stars | sailor | follows | stars || the sailor studies the night sky and uses the stars to guide the ship .
the knight crosses the moat | knight | crosses | moat || the knight lowers the drawbridge and crosses the deep castle moat .
the ranger tracks the footsteps | ranger | tracks | footsteps || the ranger follows fresh footsteps through the muddy forest path .
the adventurer lights the torch | adventurer | lights | torch || the adventurer strikes a spark and lights the torch before entering darkness .
the traveler escapes the storm | traveler | escapes | storm || the traveler spots a rocky shelter and reaches it before the heavy storm .
the explorer climbs the ridge | explorer | climbs | ridge || the explorer carefully climbs the steep ridge to survey the land ahead .
the captain saves the map | captain | saves | map || the captain catches the map before a gust of wind carries it overboard .
the scout finds the spring | scout | finds | spring || the scout follows green plants to a small freshwater spring .
the knight offers a truce | knight | offers | truce || the knight lowers his sword and offers to settle the conflict peacefully .
the traveler returns the artifact | traveler | returns | artifact || the traveler brings the ancient artifact back to the village that protected it .
the explorer discovers the passage | explorer | discovers | passage || the explorer moves a loose stone and finds a narrow passage behind the wall .
the crew lowers the anchor | crew | lowers | anchor || the crew releases the heavy anchor to keep the ship near the harbor .
the guide builds the shelter | guide | builds | shelter || the guide gathers branches and builds a small shelter before nightfall .
the sailor repairs the lantern | sailor | repairs | lantern || the sailor replaces the broken lantern glass so the deck can be seen at night .
the explorer documents the discovery | explorer | documents | discovery || the explorer records notes and sketches the unusual markings found on the rock .
""",
}

# The setting choices are category-specific to reduce nonsensical combinations.
SETTINGS = {
    "pets": ("near the house", "in the yard", "at the park", "beside the fence", "on the path", "under the trees"),
    "nature": ("in the forest", "near the river", "on the hillside", "among the trees", "at the meadow", "under the open sky"),
    "people": ("at the workplace", "in the city", "near the building", "inside the room", "in the neighborhood", "at the center"),
    "technology": ("inside the laboratory", "at the workshop", "near the station", "inside the facility", "at the research center", "in the test area"),
    "adventure": ("along the trail", "near the harbor", "in the valley", "at the campsite", "by the ancient road", "near the mountain"),
}
TIMES = ("at sunrise", "in the morning", "at midday", "in the afternoon", "at sunset")
CONDITIONS = ("during a quiet moment", "during a busy moment", "during an unexpected moment", "during a calm moment")


def lines(text):
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def originals():
    idea_lines, detail_lines = lines(BASE_IDEAS), lines(BASE_DETAILS)
    if len(idea_lines) != len(detail_lines):
        raise ValueError("The embedded original idea/detail datasets do not align.")
    category_cuts = (16, 26, 38, 46, 52)
    categories = tuple(SETTINGS)
    rows = []
    for i, (idea, detail) in enumerate(zip(idea_lines, detail_lines)):
        parts = [part.strip() for part in idea.split("|")]
        if len(parts) != 4:
            raise ValueError(f"Bad embedded idea line {i + 1}: {idea}")
        category = categories[next(j for j, end in enumerate(category_cuts) if i < end)]
        rows.append((parts[0].lower(), parts[1], parts[2], parts[3], detail.lower(), category))
    return rows


def extra_events():
    rows = []
    for event_group in (ADDITIONAL_EVENTS, EXPANDED_EVENTS):
        for category, raw in event_group.items():
            for line in lines(raw):
                left, right = line.split("||", maxsplit=1)
                parts = [part.strip() for part in left.split("|")]
                if len(parts) != 4:
                    raise ValueError(f"Bad additional example: {line}")
                rows.append((*parts, right.strip(), category))
    return rows


def optional_file_pairs(ideas_path, details_path):
    """External files are OPTIONAL; the program runs with this Python file alone."""
    if bool(ideas_path) != bool(details_path):
        raise ValueError("Provide BOTH --ideas-file and --details-file, or neither.")
    if not ideas_path or not Path(ideas_path).exists():
        return []
    idea_lines = lines(Path(ideas_path).read_text(encoding="utf-8"))
    detail_lines = lines(Path(details_path).read_text(encoding="utf-8"))
    if len(idea_lines) != len(detail_lines):
        raise ValueError(f"External file pair mismatch: {len(idea_lines)} vs {len(detail_lines)}")
    # Preserve category-specific settings for pairs already embedded.
    known_category = {row[0]: row[-1] for row in originals() + extra_events()}
    rows = []
    for idea, detail in zip(idea_lines, detail_lines):
        parts = [part.strip().lower() for part in idea.split("|")]
        if len(parts) != 4:
            raise ValueError(f"An external idea needs: sentence | who | action | what: {idea}")
        rows.append((*parts, detail.strip().lower(), known_category.get(parts[0], "people")))
    return rows


def make_example(base, place="none", when="none", condition="none"):
    situation, who, action, what, detail, _ = base
    supplement = " ".join(part for part in (place, when, condition) if part != "none")
    idea = (situation + " " + supplement).strip()
    detail = detail.strip().removesuffix(".").rstrip()
    expanded = detail + (" " + supplement if supplement else "") + " ."
    labels = (who, action, what, place, when, condition)
    return (idea, expanded, labels)


def build_examples(total, seed, external_ideas="", external_details=""):
    cores = originals() + extra_events() + optional_file_pairs(external_ideas, external_details)
    # Deduplicate base situations. User-supplied custom pairs override matching
    # built-in situations, without creating conflicting duplicated targets.
    unique = {row[0]: row for row in cores}
    cores = list(unique.values())
    base_examples = [make_example(row) for row in cores]
    generated = []
    for row in cores:
        for place, when, condition in itertools.product(SETTINGS[row[-1]], TIMES, CONDITIONS):
            generated.append(make_example(row, place, when, condition))
    random.Random(seed).shuffle(generated)
    if total < len(base_examples) + 10:
        raise ValueError(f"--examples must be at least {len(base_examples) + 10}")
    if total > len(base_examples) + len(generated):
        raise ValueError(f"--examples exceeds available unique examples ({len(base_examples) + len(generated)}).")
    selected = generated[:total - len(base_examples)]
    # Keep every source situation in training, reserve unseen recombinations for
    # validation. These are NOT a reliable test of real-world generalization.
    val_size = max(10, int(len(selected) * 0.1))
    # Repeat the original short situations in TRAINING ONLY. Without this,
    # nearly every example has appended context and bare ideas are neglected.
    train = base_examples * 10 + selected[val_size:]
    valid = selected[:val_size]
    random.Random(seed + 1).shuffle(train)
    return train, valid, len(base_examples), len(selected)


def encode_data(train, valid):
    everything = train + valid
    vocab = sorted({token for idea, detail, _ in everything for text in (idea, detail) for token in text.split()})
    words = [PAD, START, END, UNK] + vocab
    token_to_id = {v: i for i, v in enumerate(words)}
    label_names = [sorted({ex[2][n] for ex in everything}) for n in range(len(FIELDS))]
    label_to_id = [{label: i for i, label in enumerate(names)} for names in label_names]
    max_idea = max(len(idea.split()) for idea, _, _ in everything)
    max_detail = max(len(detail.split()) + 1 for _, detail, _ in everything)

    def tensors(examples):
        idea, lengths, details_x, details_y, labels = [], [], [], [], []
        for text, detail, roles in examples:
            a = [token_to_id[w] for w in text.split()]
            b = [token_to_id[START]] + [token_to_id[w] for w in detail.split()] + [token_to_id[END]]
            idea.append(a + [token_to_id[PAD]] * (max_idea - len(a)))
            lengths.append(len(a))
            details_x.append(b[:-1] + [token_to_id[PAD]] * (max_detail - len(b) + 1))
            details_y.append(b[1:] + [token_to_id[PAD]] * (max_detail - len(b) + 1))
            labels.append([label_to_id[n][roles[n]] for n in range(len(FIELDS))])
        return {
            "ideas": torch.tensor(idea, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "x": torch.tensor(details_x, dtype=torch.long),
            "y": torch.tensor(details_y, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return {
        "words": words, "word_to_id": token_to_id, "label_names": label_names,
        "train": tensors(train), "valid": tensors(valid),
        "max_idea": max_idea, "max_detail": max_detail,
        "samples": train, "validation_samples": valid,
    }


def batches(dataset, batch_size, shuffle=True):
    order = torch.randperm(len(dataset["ideas"])) if shuffle else torch.arange(len(dataset["ideas"]))
    for first in range(0, len(order), batch_size):
        index = order[first:first + batch_size]
        yield {key: value[index] for key, value in dataset.items()}


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


# =====================================================================
# 2. STAGE ONE — Transformer predicts the next word from prior words
# =====================================================================

class TransformerBlock(nn.Module):
    """Causal self-attention (future is hidden), plus a feed-forward layer."""
    def __init__(self, width):
        super().__init__()
        self.norm_a = nn.LayerNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.combine = nn.Linear(width, width)
        self.norm_b = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x):
        h = self.norm_a(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        scores = q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1])
        t = x.shape[1]
        causal_mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        attention = scores.masked_fill(causal_mask, float("-inf")).softmax(dim=-1)
        x = x + self.combine(attention @ v)
        return x + self.feed_forward(self.norm_b(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocabulary, context=36, width=64, depth=2, conditioned=False, num_layers=None, context_size=None):
        super().__init__()
        if context_size is not None:
            context = context_size
        if num_layers is not None:
            depth = num_layers
        self.context = context
        self.context_size = context
        self.width = width
        self.words = nn.Embedding(vocabulary, width)
        self.positions = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([TransformerBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.predict = nn.Linear(width, vocabulary)
        self.idea_projection = nn.Linear(width, width) if conditioned else None
        self.idea_to_words = self.idea_projection

    def features(self, ids, idea=None):
        if ids.shape[1] > self.context:
            raise ValueError("Sequence exceeds the configured context window")
        t = ids.shape[1]
        x = self.words(ids) + self.positions(torch.arange(t, device=ids.device))
        if self.idea_projection is not None:
            if idea is None:
                raise ValueError("The detail model requires an idea vector")
            if idea.dim() == 2:
                x = x + self.idea_projection(idea)[:, None, :]
            else:
                x = x + self.idea_projection(idea)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward_features(self, ids, idea=None):
        return self.features(ids, idea)

    def forward(self, ids, idea=None):
        return self.predict(self.features(ids, idea))


# =====================================================================
# 3. STAGE TWO — Read a situation, compress it to an idea vector
# =====================================================================

class IdeaReader(nn.Module):
    """GRU turns a variable-length sentence into one vector and six labels."""
    def __init__(self, vocabulary, width=64, label_names=None, pad_id=0):
        super().__init__()
        if isinstance(width, (list, tuple)) and label_names is not None:
            pad_id = label_names
            label_names = width
            width = 64
        self.width = width
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=pad_id)
        self.reader = nn.GRU(width, width, batch_first=True)
        self.heads = nn.ModuleList(nn.Linear(width, len(names)) for names in label_names)
        if len(label_names) >= 3:
            self.who = self.heads[0]
            self.action = self.heads[1]
            self.what = self.heads[2]

    def forward(self, ids, lengths=None):
        # Read right-padded sentences; select the output at the final REAL
        # word. This avoids special packed-sequence operations on Mac MPS.
        states, _ = self.reader(self.embedding(ids))
        if lengths is None:
            idea = states[:, -1, :]
        else:
            last_position = (lengths.to(ids.device) - 1).clamp_min(0)
            idea = states[torch.arange(ids.size(0), device=ids.device), last_position]
        return idea, [head(idea) for head in self.heads]


# =====================================================================
# 4. STAGE THREE — Generate detailed words conditioned on an idea
# =====================================================================

# Same Transformer code as Stage One, but conditioned=True means it receives
# the SINGLE vector from Stage Two. It never reads the original situation.
# Teacher forcing during training: actual previous detailed words predict next.


# =====================================================================
# 5. TRAINING — Batched learning, validation, checkpoints, and resuming
# =====================================================================

def word_loss(logits, targets, pad_id):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id)


def batch_loss(stage, model, reader, batch, pad_id):
    if stage == 1:
        return word_loss(model(batch["x"]), batch["y"], pad_id)
    if stage == 2:
        _, predictions = reader(batch["ideas"], batch["lengths"])
        return sum(F.cross_entropy(pred, batch["labels"][:, n]) for n, pred in enumerate(predictions)) / len(FIELDS)
    if stage == 3:
        with torch.no_grad():
            idea, _ = reader(batch["ideas"], batch["lengths"])
        return word_loss(model(batch["x"], idea), batch["y"], pad_id)
    raise ValueError(f"Unknown training stage {stage}")


def validation(stage, model, reader, valid, pad_id, device, batch_size, ablate=False):
    model.eval()
    reader.eval()
    losses, sizes, correct, possible, swapped = [], [], 0, 0, []
    with torch.no_grad():
        for source in batches(valid, batch_size, shuffle=False):
            batch = to_device(source, device)
            n = len(batch["ideas"])
            loss = batch_loss(stage, model, reader, batch, pad_id)
            losses.append(loss.item() * n)
            sizes.append(n)
            if stage == 2:
                _, predictions = reader(batch["ideas"], batch["lengths"])
                correct += sum(int((p.argmax(-1) == batch["labels"][:, j]).sum()) for j, p in enumerate(predictions))
                possible += n * len(FIELDS)
            if stage == 3 and ablate and n > 1:
                vectors, _ = reader(batch["ideas"], batch["lengths"])
                shifted_vectors = vectors.roll(1, dims=0)
                swapped.append((word_loss(model(batch["x"], shifted_vectors), batch["y"], pad_id).item(), n))
    value = sum(losses) / sum(sizes) if sizes else 0.0
    accuracy = correct / possible if possible else None
    shuffled = sum(x * n for x, n in swapped) / sum(n for _, n in swapped) if swapped else None
    return value, accuracy, shuffled


def train_stage(stage, model, reader, data, args, device):
    name = {1: "WORD PREDICTION", 2: "SYNTHETIC ROLE-LABEL FIT", 3: "IDEA-CONDITIONED WRITING"}[stage]
    pad_id = data["word_to_id"][PAD]
    print(f"\n=== STAGE {stage}: {name} ===", flush=True)
    epochs = args.idea_epochs if stage == 2 else args.epochs
    model.train()
    if stage == 3:
        reader.eval()  # Freeze the trained idea reader as an explicit bottleneck.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    updates = 0
    if args.steps == 0 or epochs == 0:
        print("  Training skipped (--steps 0 or --epochs 0).")
    else:
        for epoch in range(1, epochs + 1):
            total_loss, total_items = 0.0, 0
            for source in batches(data["train"], args.batch_size):
                if args.steps is not None and updates >= args.steps:
                    break
                batch = to_device(source, device)
                model.train()
                loss = batch_loss(stage, model, reader, batch, pad_id)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                count = len(batch["ideas"])
                total_loss += loss.item() * count
                total_items += count
                updates += 1
            if total_items:
                valid_loss, acc, _ = validation(stage, model, reader, data["valid"], pad_id, device, args.batch_size)
                extra = f" | synthetic-label accuracy {acc:.1%}" if acc is not None else ""
                print(f"  epoch {epoch}/{epochs} | updates {updates} | train loss {total_loss / total_items:.3f} | validation loss {valid_loss:.3f}{extra}", flush=True)
            if args.steps is not None and updates >= args.steps:
                break
    v, a, shuffled = validation(stage, model, reader, data["valid"], pad_id, device, args.batch_size, ablate=stage == 3)
    print(f"  final validation loss: {v:.3f}" + (f" | synthetic-label accuracy: {a:.1%}" if a is not None else ""))
    if shuffled is not None:
        print(f"  conditioning ablation: wrong/shuffled idea loss {shuffled:.3f} vs correct-idea loss {v:.3f}")
        print("  (If losses are similar, the writer is not reliably using its idea signal.)")
    return v


def save_checkpoint(path, data, args, word_model, reader, writer):
    snapshot = {
        "words": data["words"], "label_names": data["label_names"],
        "architecture": (args.width, args.layers, data["max_detail"]),
        "word_model": word_model.state_dict(), "reader": reader.state_dict(),
        "writer": writer.state_dict(),
        "run_config": {k: getattr(args, k, None) for k in
                       ("examples", "seed", "width", "layers", "ideas_file", "details_file")},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Atomic checkpoint replacement protects the previous learning state if a
    # save is interrupted. No optimizer momentum is persisted in this version.
    temporary = str(path) + ".tmp"
    try:
        torch.save(snapshot, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"\nSaved all three stages to: {path}")


def load_checkpoint(path, data, args, word_model, reader, writer, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if (state["words"][:len(data["words"])] != data["words"] or state["label_names"] != data["label_names"]
            or tuple(state["architecture"]) != (args.width, args.layers, data["max_detail"])):
        raise ValueError("Checkpoint incompatible with original dataset or architecture. Reuse --examples/--seed/--width/--layers.")
    # The original training vocabulary remains a stable prefix; approved personal
    # learning adds vocabulary at the end without renumbering original tokens.
    if len(state["words"]) > len(data["words"]):
        extend_vocabulary(state["words"][len(data["words"]):], data, word_model, reader, writer)
    word_model.load_state_dict(state["word_model"])
    reader.load_state_dict(state["reader"])
    writer.load_state_dict(state["writer"])
    print(f"Restored prior learned weights from: {path}")


# =====================================================================
# 6. INSPECTION — Predictions, meanings, generation, and idea-signal checks
# =====================================================================

def encode_idea(text, data, device):
    words = text.lower().strip().split()
    if not words:
        raise ValueError("Idea cannot be empty")
    unknown = sorted(set(words) - set(data["word_to_id"]))
    if unknown:
        raise ValueError(f"Words not in the trained vocabulary: {unknown}")
    if len(words) > data["max_idea"]:
        raise ValueError(f"Idea longer than trained window ({data['max_idea']} words)")
    ids = [data["word_to_id"][w] for w in words]
    return torch.tensor([ids], device=device), torch.tensor([len(ids)])


def inspect_idea(reader, text, data, device):
    ids, lengths = encode_idea(text, data, device)
    reader.eval()
    with torch.no_grad():
        vector, pred = reader(ids, lengths)
    labels = {field: data["label_names"][i][p.argmax(-1).item()] for i, (field, p) in enumerate(zip(FIELDS, pred))}
    return vector, labels


def next_words(model, prompt, data, device, top=5):
    ids = [data["word_to_id"].get(w, data["word_to_id"][UNK]) for w in prompt.lower().split()]
    if not ids:
        ids = [data["word_to_id"][START]]
    ids = ids[-model.context:]
    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor([ids], device=device))[0, -1].softmax(-1)
        values, indices = scores.topk(top)
    return [(data["words"][int(i)], round(float(v), 3)) for v, i in zip(values, indices)]


def generate(writer, vector, data, device, trace=False, max_words=36):
    writer.eval()
    ids = [data["word_to_id"][START]]
    forbidden = [data["word_to_id"][PAD], data["word_to_id"][START], data["word_to_id"][UNK]]
    with torch.no_grad():
        for step in range(max_words):
            x = torch.tensor([ids[-writer.context:]], device=device)
            logits = writer(x, vector)[0, -1].clone()
            logits[forbidden] = -float("inf")
            probabilities = logits.softmax(-1)
            scores, candidates = probabilities.topk(3)
            next_token = int(candidates[0])
            if trace:
                print(f"  token {step + 1:02}: " + ", ".join(f"{data['words'][int(idx)]}={float(prob):.2f}" for prob, idx in zip(scores, candidates)))
            if next_token == data["word_to_id"][END]:
                break
            ids.append(next_token)
    return " ".join(data["words"][i] for i in ids[1:]).replace(" .", ".")


# =====================================================================
# 7. OPTIONAL STAGE FOUR — Quick adaptation vs retained older ability
# =====================================================================

class FastSlowModel(nn.Module):
    """Frozen core with a gated low-rank adapter; gates may leak to old ideas."""
    def __init__(self, base, key):
        super().__init__()
        self.base = base
        self.key = key.detach()
        self.down = nn.Linear(base.width, 12, bias=False)
        self.up = nn.Linear(12, base.predict.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, words, idea):
        with torch.no_grad():
            features = self.base.features(words, idea)
            old = self.base.predict(features)
        sim = F.cosine_similarity(idea, self.key, dim=-1)
        gate = torch.sigmoid((sim - 0.80) * 20).view(-1, 1, 1)
        return old + gate * self.up(self.down(features))


def custom_target(detail, data, device):
    tokens = detail.lower().split()
    unseen = sorted(set(tokens) - set(data["word_to_id"]))
    if unseen:
        raise ValueError(f"Unknown target words for adaptation: {unseen}")
    seq = [data["word_to_id"][START]] + [data["word_to_id"][w] for w in tokens] + [data["word_to_id"][END]]
    if len(seq) - 1 > data["max_detail"]:
        raise ValueError("Custom target is longer than trained context")
    return torch.tensor([seq[:-1]], device=device), torch.tensor([seq[1:]], device=device)


def demonstrate_adaptation(writer, reader, data, args, device):
    print("\n=== STAGE 4 (OPTIONAL): FAST VS SLOW ADAPTATION ===")
    idea, labels = inspect_idea(reader, args.novel_idea, data, device)
    x, y = custom_target(args.novel_detail, data, device)
    pad = data["word_to_id"][PAD]
    old_sample = to_device(next(batches(data["valid"], min(48, len(data["valid"])), shuffle=False)), device)
    with torch.no_grad():
        old_ideas, _ = reader(old_sample["ideas"], old_sample["lengths"])

    def old_loss(model):
        with torch.no_grad():
            return word_loss(model(old_sample["x"], old_ideas), old_sample["y"], pad).item()

    def new_loss(model):
        with torch.no_grad():
            return word_loss(model(x, idea), y, pad).item()

    writer.eval()
    base_old, base_new = old_loss(writer), new_loss(writer)
    naive = copy.deepcopy(writer)
    opt = torch.optim.AdamW(naive.parameters(), lr=0.003)
    for _ in range(args.adapt_steps):
        loss = word_loss(naive(x, idea), y, pad)
        opt.zero_grad()
        loss.backward()
        opt.step()
    naive.eval()

    fast = FastSlowModel(writer, idea).to(device)
    opt_fast = torch.optim.Adam(fast.parameters(), lr=0.02)
    for _ in range(args.adapt_steps):
        loss = word_loss(fast(x, idea), y, pad)
        opt_fast.zero_grad()
        loss.backward()
        opt_fast.step()
    fast.eval()
    print(f"  novel: {args.novel_idea}  =>  {args.novel_detail}")
    print("  Method             Old-example loss   New-example loss")
    print(f"  Before adaptation       {base_old:7.3f}            {base_new:7.3f}")
    print(f"  Full-weight update      {old_loss(naive):7.3f}            {new_loss(naive):7.3f}")
    print(f"  Gated fast adapter      {old_loss(fast):7.3f}            {new_loss(fast):7.3f}")
    print("  NOTE: Lower is better. Frozen core != zero forgetting: an adapter")
    print("  can still interfere with older predictions, especially for similar ideas.")


# =====================================================================
# 8. EXPERIENTIAL RECORDS — Local evidence, provenance, and approvals
# =====================================================================

# JSONL is append-only. An observation never becomes a fact merely because
# the model echoed it. Data and model outputs are separately identified.
# NOTE: These files are intentionally NOT read by weights-only evaluation.


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def append_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Restricted permissions on POSIX; Windows permissions depend on the folder.
    fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def candidate_fingerprint(row):
    content = json.dumps([row.get("question"), row.get("answer"), row.get("split")], ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def has_revoked_or_corrected_training(data_dir):
    records = read_records(Path(data_dir) / "learning_events.jsonl")
    current = {r["id"]: r for r in latest_candidates(data_dir)}
    for record in records:
        row = current.get(record["candidate_id"])
        if (row is None or row["status"] != "approved" or row["split"] != "train"
                or record.get("fingerprint") != candidate_fingerprint(row)):
            return True
    return False


def latest_candidates(data_dir):
    # The newest revision wins; history and rejections remain auditable.
    candidates = {}
    for row in read_records(Path(data_dir) / "candidates.jsonl"):
        candidates[row["id"]] = row
    return list(candidates.values())


def personal_tokens(value):
    # Stable word-level tokenizer compatible with original space-separated
    # training examples ("ball ." remains two tokens).
    return re.findall(r"[\w]+(?:['-][\w]+)*|[^\w\s]", value.lower(), flags=re.UNICODE)


def create_candidate(data_dir, evidence, question="", answer="", split="train"):
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    if evidence.get("source") != "user":
        raise ValueError("Only an actual user's observation can become evidence")
    candidate = {
        "id": uuid.uuid4().hex[:12], "created_at": now_iso(),
        "evidence_id": evidence["id"], "original_observation": evidence["text"],
        "session_id": evidence["session_id"], "event_index": evidence["event_index"],
        "event_time": evidence["timestamp"], "source": evidence["source"],
        "question": question, "answer": answer, "split": split,
        "status": "pending", "origin": "human-review-scaffold",
        "interpretation": {k: None for k in ("WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW", "DETAILS", "IDEA")},
        "uncertainty": "Unreviewed; no automatic semantic interpretation is asserted.",
    }
    append_record(Path(data_dir) / "candidates.jsonl", candidate)
    return candidate


def show_review(data_dir, interactive=True):
    pending = [row for row in latest_candidates(data_dir) if row["status"] in ("pending", "deferred")]
    if not pending:
        print("No pending candidates.")
        return
    for row in pending:
        print(f"\nCandidate {row['id']} from {row['event_time']} [{row['session_id']} event {row['event_index']}]")
        print("  Evidence:", row["original_observation"])
        print("  Proposed Q:", row["question"] or "(none; user must author)")
        print("  Proposed A:", row["answer"] or "(none; user must author)")
        print("  Status: no model-derived fact is verified yet")
        if not interactive:
            continue
        decision = input("  [a]pprove / [e]dit / [t]est holdout / [r]eject / [d]efer / [b]atch ready / [q]uit: ").strip().lower()
        if decision == "q":
            return
        if decision == "b":
            count = 0
            for candidate in pending:
                if personal_tokens(candidate["question"]) and personal_tokens(candidate["answer"]):
                    append_record(Path(data_dir) / "candidates.jsonl",
                                  dict(candidate, status="approved", updated_at=now_iso()))
                    count += 1
            print(f"  Batch-approved {count} PRELABELED candidates; blank scaffolds remain pending")
            return
        if decision not in ("a", "e", "t", "r", "d"):
            continue
        revised = dict(row)
        if decision in ("e", "t") or (decision == "a" and not row["question"].strip()):
            revised["question"] = input("  Ask a precise question about this evidence: ").strip()
            revised["answer"] = input("  Verified short answer (supported by evidence): ").strip()
        if decision == "t":
            revised["split"] = "test"
        if decision in ("a", "e", "t"):
            if not personal_tokens(revised["question"]) or not personal_tokens(revised["answer"]):
                print("  Both question and answer required; remains pending")
                continue
            if len(personal_tokens(revised["answer"])) > 40:
                print("  Keep the answer short; remains pending")
                continue
            if decision == "e":
                revised["status"] = "pending"  # Editing NEVER equals authorization.
                print("  Edit saved as pending. Review again and explicitly approve.")
            else:
                revised["status"] = "approved"
                print("  Approval applies to a human-reviewed label, NOT a verified model inference.")
        else:
            revised["status"] = "rejected" if decision == "r" else "deferred"
        revised["updated_at"] = now_iso()
        append_record(Path(data_dir) / "candidates.jsonl", revised)
        print("  Saved:", revised["status"], "split:", revised["split"])


def revise_candidate(data_dir, candidate_id):
    records = {r["id"]: r for r in latest_candidates(data_dir)}
    if candidate_id not in records:
        raise ValueError(f"Candidate {candidate_id!r} not found")
    old = records[candidate_id]
    print("Evidence (immutable):", old["original_observation"])
    print("Current status:", old["status"], "Q:", old["question"], "A:", old["answer"])
    choice = input("[c]orrect Q/A / [r]eject / [d]efer / [q]uit: ").strip().lower()
    if choice not in ("c", "r", "d"):
        return
    new = dict(old)
    if choice == "c":
        new["question"] = input("Corrected question: ").strip()
        new["answer"] = input("Corrected answer (only supported evidence): ").strip()
        if not personal_tokens(new["question"]) or not personal_tokens(new["answer"]):
            raise ValueError("Empty corrected Q/A")
        new["status"] = "approved" if input("Approve corrected Q/A for learning? [y/N]: ").strip().lower() == "y" else "pending"
    elif choice == "r":
        new["status"] = "rejected"
    else:
        new["status"] = "deferred"
    new["updated_at"] = now_iso()
    append_record(Path(data_dir) / "candidates.jsonl", new)
    print("Revision saved. If this candidate was previously trained, run --mode learn to rebuild personal weights.")


# =====================================================================
# 9. PLASTICITY — Extend word representations and update the SAME models
# =====================================================================


def extend_vocabulary(new_words, data, word_model, reader, writer):
    # New names such as Jake cannot be learned if everything is mapped to UNK.
    # Preserve all old token IDs and existing weights when extending the vocab.
    additions = [w for w in new_words if w not in data["word_to_id"]]
    if not additions:
        return 0
    previous = len(data["words"])
    enlarged = previous + len(additions)
    data["words"].extend(additions)
    data["word_to_id"].update({w: i for i, w in enumerate(data["words"])})
    for embedding_owner, key in ((reader, "embedding"), (word_model, "words"), (writer, "words")):
        original = getattr(embedding_owner, key)
        e = nn.Embedding(enlarged, original.embedding_dim, padding_idx=original.padding_idx).to(original.weight.device)
        with torch.no_grad():
            e.weight[:previous].copy_(original.weight)
            e.weight[previous:].copy_(original.weight[data["word_to_id"][UNK]].unsqueeze(0).expand(len(additions), -1))
            e.weight[previous:].add_(torch.randn_like(e.weight[previous:]) * 0.01)
        setattr(embedding_owner, key, e)
    for model in (word_model, writer):
        output = model.predict
        new_output = nn.Linear(output.in_features, enlarged).to(output.weight.device)
        with torch.no_grad():
            new_output.weight[:previous].copy_(output.weight)
            new_output.bias[:previous].copy_(output.bias)
            new_output.weight[previous:].copy_(output.weight[data["word_to_id"][UNK]].unsqueeze(0).expand(len(additions), -1))
            new_output.bias[previous:].fill_(-4.0)
        model.predict = new_output
    return len(additions)


def qa_tensors(question, answer, data, device):
    q, a = personal_tokens(question), personal_tokens(answer)
    if not q or not a:
        raise ValueError("Question and answer must be nonempty")
    if len(a) + 1 > data["max_detail"]:
        raise ValueError(f"Answer exceeds {data['max_detail'] - 1} tokens; split it into shorter facts")
    ids = data["word_to_id"]
    # Reader is a GRU, so it can read longer question sequences without a
    # fixed position embedding, unlike the causal language decoder.
    question_ids = torch.tensor([[ids.get(w, ids[UNK]) for w in q]], device=device)
    answer_ids = [ids[START]] + [ids.get(w, ids[UNK]) for w in a] + [ids[END]]
    return (question_ids, torch.tensor([len(q)], device=device),
            torch.tensor([answer_ids[:-1]], device=device),
            torch.tensor([answer_ids[1:]], device=device))


def personal_loss(reader, writer, question, answer, data, device):
    q, length, x, y = qa_tensors(question, answer, data, device)
    idea, _ = reader(q, length)
    return word_loss(writer(x, idea), y, data["word_to_id"][PAD])


def answer_with_weights(reader, writer, question, data, device):
    tokens = personal_tokens(question)
    if not tokens:
        raise ValueError("Question cannot be blank")
    q = torch.tensor([[data["word_to_id"].get(t, data["word_to_id"][UNK]) for t in tokens]], device=device)
    reader.eval()
    writer.eval()
    with torch.no_grad():
        vector, _ = reader(q, torch.tensor([len(tokens)], device=device))
        # Generation does not consult any previous event or personal file.
        return generate(writer, vector, data, device, max_words=min(22, data["max_detail"]))


def personal_evaluation(rows, reader, writer, data, device):
    reader.eval()
    writer.eval()
    scores = []
    with torch.no_grad():
        for row in rows:
            try:
                scores.append(float(personal_loss(reader, writer, row["question"], row["answer"], data, device)))
            except ValueError:
                continue
    return sum(scores) / len(scores) if scores else None


def conditioning_check(rows, reader, writer, data, device):
    # Does a mismatched question predict a target nearly as well? If yes,
    # memorized target style may explain the gain rather than Q->A learning.
    if len(rows) < 2:
        return None
    reader.eval()
    writer.eval()
    consistent, mismatched = [], []
    with torch.no_grad():
        for index, row in enumerate(rows):
            other = rows[(index + 1) % len(rows)]
            correct_q, q_len, target_x, target_y = qa_tensors(row["question"], row["answer"], data, device)
            wrong_q, wrong_len, _, _ = qa_tensors(other["question"], row["answer"], data, device)
            vector, _ = reader(correct_q, q_len)
            swapped_vector, _ = reader(wrong_q, wrong_len)
            consistent.append(float(word_loss(writer(target_x, vector), target_y, data["word_to_id"][PAD])))
            mismatched.append(float(word_loss(writer(target_x, swapped_vector), target_y, data["word_to_id"][PAD])))
    return (sum(consistent) / len(consistent), sum(mismatched) / len(mismatched))


def learn_approved(data_dir, args, data, word_model, reader, writer, device):
    rows = [r for r in latest_candidates(data_dir) if r["status"] == "approved"]
    train = [r for r in rows if r["split"] == "train"]
    holdouts = [r for r in rows if r["split"] == "test"]
    if not train:
        print("No approved training examples. Nothing changed.")
        return False
    # Fail before touching weights if any material exceeds decoder context.
    for r in train + holdouts:
        if len(personal_tokens(r["answer"])) + 1 > data["max_detail"]:
            raise ValueError(f"{r['id']}: answer too long for current decoder; shorten it")
    lexicon = sorted({t for r in train for key in ("question", "answer") for t in personal_tokens(r[key])})
    added = extend_vocabulary(lexicon, data, word_model, reader, writer)
    print(f"Approved: {len(train)} train / {len(holdouts)} held out; {added} new vocabulary tokens")
    # Holdout vocabulary is deliberately never added from holdout examples.
    from_train = lambda: personal_evaluation(train, reader, writer, data, device)
    before = from_train()
    held_before = personal_evaluation(holdouts, reader, writer, data, device)
    old_baseline, _, _ = validation(3, writer, reader, data["valid"], data["word_to_id"][PAD], device, args.batch_size)
    original_reader = copy.deepcopy(reader.state_dict())
    original_writer = copy.deepcopy(writer.state_dict())
    # This is real gradient learning: train the existing idea encoder + writer.
    reader.train()
    writer.train()
    optimizer = torch.optim.AdamW(list(reader.parameters()) + list(writer.parameters()), lr=args.personal_lr)
    # Revisit a rotating set of older public examples while learning new facts.
    # This is genuine gradient-based replay, not prompt-time memory retrieval.
    replay_stream = iter(batches(data["train"], min(8, args.batch_size)))
    for step in range(args.personal_steps):
        r = train[step % len(train)]
        loss = personal_loss(reader, writer, r["question"], r["answer"], data, device)
        if args.replay_weight:
            try:
                sample = next(replay_stream)
            except StopIteration:
                replay_stream = iter(batches(data["train"], min(8, args.batch_size)))
                sample = next(replay_stream)
            replay = to_device(sample, device)
            replay_idea, replay_labels = reader(replay["ideas"], replay["lengths"])
            old_language = word_loss(writer(replay["x"], replay_idea), replay["y"], data["word_to_id"][PAD])
            # Keep the reader's original meaning labels grounded too.
            old_meaning = sum(F.cross_entropy(p, replay["labels"][:, j]) for j, p in enumerate(replay_labels)) / len(FIELDS)
            loss = loss + args.replay_weight * (old_language + 0.2 * old_meaning)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(reader.parameters()) + list(writer.parameters()), 1.0)
        optimizer.step()
    after = from_train()
    held_after = personal_evaluation(holdouts, reader, writer, data, device)
    swapped = conditioning_check(train, reader, writer, data, device)
    old_after, _, _ = validation(3, writer, reader, data["valid"], data["word_to_id"][PAD], device, args.batch_size)
    accepted = after < before and old_after <= old_baseline + args.max_retention_loss_rise
    print(f"  Approved-target token loss: {before:.3f} -> {after:.3f}")
    print(f"  Original held-out detail loss: {old_baseline:.3f} -> {old_after:.3f}")
    if held_before is not None:
        print(f"  Manually held-out Q/A loss: {held_before:.3f} -> {held_after:.3f} (not trained)")
    if swapped:
        print(f"  Question conditioning: correct Q loss {swapped[0]:.3f}; swapped Q loss {swapped[1]:.3f}")
        if swapped[1] <= swapped[0] + 0.05:
            print("  WARNING: The decoder may be ignoring the question; target memorization is not association.")
    else:
        print("  Question conditioning UNTESTED: supply at least two different approved Q/A pairs.")
    if not accepted:
        reader.load_state_dict(original_reader)
        writer.load_state_dict(original_writer)
        print("  REJECTED neural update: no target gain or too much original-task degradation")
    else:
        print("  ACCEPTED neural update. Retention threshold is a weak safeguard, NOT proof of no forgetting.")
        for r in train:
            append_record(Path(data_dir) / "learning_events.jsonl", {
                "candidate_id": r["id"], "trained_at": now_iso(), "steps": args.personal_steps,
                "fingerprint": candidate_fingerprint(r),
                "method": "joint-reader-writer-gradient-finetune-with-replay", "test_split": False,
            })
    append_record(Path(data_dir) / "learning_metrics.jsonl", {
        "time": now_iso(), "approved_count": len(train), "holdout_count": len(holdouts),
        "new_tokens": added, "steps": args.personal_steps,
        "approved_loss_before": before, "approved_loss_after": after,
        "general_loss_before": old_baseline, "general_loss_after": old_after,
        "heldout_loss_before": held_before, "heldout_loss_after": held_after,
        "correct_question_loss": swapped[0] if swapped else None,
        "swapped_question_loss": swapped[1] if swapped else None,
        "accepted": accepted,
    })
    return accepted


# =====================================================================
# 10. INTERACTIVE SESSION — Explicit attribution; no self-generated facts
# =====================================================================


def run_session(data_dir, enable_research=False, research_model="qwen2.5:1.5b-instruct",
                enable_world_model=False, world_checkpoint="", enable_adaptive_loop=False,
                enable_learned_sparse=False, sparse_budget=8):
    session_id = uuid.uuid4().hex[:12]
    from sonara_memory import SonaraMemorySystem
    from sonara_research import ResearchCognition
    from sonara_world_model import SonaraWorldModel
    from sonara_adaptive import AdaptiveLoop, cue_tokens, overlap
    mem_sys = SonaraMemorySystem(Path(data_dir) / "sonara_memories")
    research = ResearchCognition(data_dir, local_model=research_model) if enable_research else None
    world_model = None
    if enable_world_model:
        world_ckpt = Path(world_checkpoint) if world_checkpoint else (Path(data_dir) / "world_model.pt")
        try:
            world_model = SonaraWorldModel(checkpoint_path=world_ckpt if world_ckpt.exists() else None, device="cpu")
            world_model._learned_sparse_activation = bool(enable_learned_sparse)
            world_model._max_active_events = sparse_budget
            # Reconstitute the working view from canonical evidence after a restart.
            # Model weights and retrieved text are not competing memory authorities.
            for observed in mem_sys.memories.values():
                if observed.source == "user_observation" and observed.content:
                    world_model.observe(observed.content)
        except Exception as error:
            print(f"[Sonara] Warning: world model checkpoint not loaded: {error}")
            world_model = None
    pending_expectation = None
    last_research = None
    adaptive = AdaptiveLoop(read_records(Path(data_dir) / "events.jsonl") if enable_adaptive_loop else ())
    last_neural = None

    print(f"Sonara Session {session_id} | /quit | /ask <question> | /remember <text> | /recall <query> | /timeline")
    if enable_world_model and world_model is not None:
        print("World reasoning: NEURAL WORLD MODEL ACTIVE (pointer-generator state tracker).")
        if enable_learned_sparse:
            print(f"Learned sparse attention: RESEARCH ONLY, budget={sparse_budget}, "
                  f"verified selector updates={world_model._sparse_training_updates}; "
                  "untrained rankers use full context.")
    else:
        print("World reasoning: NOT IMPLEMENTED. No programmed location/temporal answers or auto-trained world classifier.")
    print("Local research: /research <question> and /rate-research <0..1> (requires --enable-research and Ollama).")
    if enable_adaptive_loop:
        print("Adaptive gate: ON. /rate-answer <0..1> rates the immediately preceding neural prediction.")
        print("The gate learns caution, not facts or neural world-model weights.")
    append_record(Path(data_dir) / "sessions.jsonl", {"id": session_id, "type": "start", "timestamp": now_iso(), "local_user": "owner"})
    event_index = 0
    while True:
        try:
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "/quit"
        if line.lower() in ("/quit", "/exit"):
            break
        if not line:
            continue
        if line == "/train-world":
            if enable_world_model and world_model is not None:
                approved = [r for r in latest_candidates(data_dir) if r.get("status") == "approved" and r.get("question") and r.get("answer")]
                if not approved:
                    print("Sonara> No approved training candidates found in candidates.jsonl.")
                    continue
                trained = 0
                for r in approved:
                    ev_list = [r.get("original_observation", "")]
                    world_model.learn_step(ev_list, r["question"], r["answer"])
                    trained += 1
                world_model.save_checkpoint()
                print(f"Sonara> Neural World Model trained on {trained} candidate(s); checkpoint saved.")
            else:
                print("Sonara> Retired: rule-labeled classifier was not evidence of world learning. No weights changed.")
            continue
        if line.startswith("/predict "):
            if world_model is None:
                print("Sonara> World model is not enabled. Start with --enable-world-model.")
                continue
            pred_q = line[len("/predict "):].strip()
            pred = world_model.predict(pred_q)
            print(f"Sonara> NEURAL PREDICTION: {pred.answer} (confidence={pred.confidence:.2f}, uncertainty={pred.uncertainty:.2f})")
            continue
        if line.startswith("/research "):
            if research is None:
                print("Sonara> Local research is disabled. Restart with --enable-research to opt in.")
                continue
            question = line[len("/research "):].strip()
            try:
                result = research.research(question)
            except (ValueError, OSError, KeyError, IndexError) as error:
                print("Sonara> Research unavailable:", type(error).__name__, str(error)[:200])
                continue
            last_research = result
            append_record(Path(data_dir) / "research_events.jsonl",
                          {"timestamp": now_iso(), "session_id": session_id,
                           "source": "local_model", **result})
            print(f"Sonara> UNVERIFIED LOCAL MODEL ({result['model']}): {result['answer']}")
            continue
        if line.startswith("/rate-research "):
            if research is None or last_research is None:
                print("Sonara> No research result to evaluate.")
                continue
            try:
                utility = float(line[len("/rate-research "):].strip())
                metrics = research.feedback(last_research["question"], utility)
            except ValueError as error:
                print("Sonara> Invalid usefulness score:", error)
                continue
            append_record(Path(data_dir) / "research_feedback.jsonl",
                          {"timestamp": now_iso(), "session_id": session_id,
                           "model": last_research["model"], "question": last_research["question"],
                           "utility": utility, "training": metrics})
            last_research = None
            print("Sonara> Local research utility feedback:", json.dumps(metrics, sort_keys=True))
            continue
        if line.startswith("/rate-answer "):
            if not enable_adaptive_loop or last_neural is None:
                print("Sonara> No neural prediction to rate. Enable --enable-adaptive-loop and --enable-world-model.")
                continue
            try:
                rating = float(line[len("/rate-answer "):].strip())
                result = adaptive.feedback(last_neural["question"], rating)
            except ValueError as error:
                print("Sonara> Invalid answer rating:", error)
                continue
            append_record(Path(data_dir) / "events.jsonl", {
                "id": uuid.uuid4().hex[:12], "source": "user", "timestamp": now_iso(),
                "session_id": session_id, "event_index": event_index,
                "type": "adaptive_feedback", "target_type": "neural_prediction",
                "question": last_neural["question"], "response_id": last_neural["response_id"],
                "rating": rating,
            })
            last_neural = None
            print(f"Sonara> Decision caution updated: {result}. No world facts or weights changed.")
            continue
        if line.startswith("/feedback "):
            print("Sonara> No world-model prediction exists to correct. "
                  "Use /remember <observation> or approve supervised Q/A in --mode review.")
            continue
        if line.startswith("/remember "):
            text = line[len("/remember "):].strip()
            mem = mem_sys.remember(text, category="episodic", source="user_observation", session_id=session_id)
            if world_model is not None:
                world_model.observe(text)
            print(f"Sonara> Memory saved: [{mem.timestamp_human}] ({mem.relative_time()})")
            print(f"        ID: {mem.id} | Entities: {mem.entities}")
            continue
        if line.startswith("/recall "):
            query = line[len("/recall "):].strip()
            matches = mem_sys.recall(query)
            if not matches:
                print("Sonara> No matching memories found.")
            else:
                print(f"Sonara> Found {len(matches)} relevant memory(s):")
                for m, score in matches:
                    print(f"  [{score:.2f}] {m.timestamp_human} ({m.relative_time()}) [{m.category}] {m.content}")
            continue
        if line.strip() == "/timeline":
            items = mem_sys.get_timeline(limit=15)
            if not items:
                print("Sonara> No memories recorded yet.")
            else:
                print("Sonara> === MEMORY TIMELINE ===")
                for it in items:
                    print(f"  • {it['timestamp_human']} ({it['relative_time']}) [{it['category']}] {it['content']}")
            continue
        if line.strip() == "/reflect":
            print("Sonara> Disabled: repeated words are not learned insights. "
                  "Legacy records remain archived, not promoted to verified knowledge.")
            continue

        event_index += 1
        last_neural = None  # ratings never attach to an earlier, unrelated turn
        ask = line.startswith("/ask ") or line.endswith("?")
        if line.startswith("/ask "):
            line = line[5:].strip()
        record = {"id": uuid.uuid4().hex[:12], "source": "user", "session_id": session_id,
                  "timestamp": now_iso(), "event_index": event_index, "text": line, "type": "question" if ask else "observation"}
        append_record(Path(data_dir) / "events.jsonl", record)
        if not ask:
            mem = mem_sys.remember(line, category="episodic", source="user_observation", session_id=session_id)
            create_candidate(data_dir, record)
            if world_model is not None:
                if pending_expectation is not None:
                    # Calculate prediction error against verified observation
                    pred_err = 0.0 if pending_expectation.get("expected_entity", "").lower() in line.lower() else 1.0
                    append_record(Path(data_dir) / "events.jsonl", {
                        "id": uuid.uuid4().hex[:12], "source": "world_model", "timestamp": now_iso(),
                        "session_id": session_id, "event_index": event_index,
                        "type": "prediction_verification",
                        "expectation": pending_expectation,
                        "observed": line,
                        "error": pred_err,
                    })
                    pending_expectation = None
                world_model.observe(line)
            response = (f"Observation stored at {mem.timestamp_human} ({mem.relative_time()}). "
                        "Optional supervised Q/A candidate created; NO neural training occurred.")
        else:
            # Adaptive mode reuses the existing memory owner for bounded retrieval,
            # never converts an observed sentence into an inferred answer.
            evidence = []
            if enable_adaptive_loop:
                matches = mem_sys.recall(line, limit=3, min_score=0.35)
                evidence = [(m, score) for m, score in matches
                            if m.source in ("user_observation", "user_explicit")
                            and overlap(cue_tokens(line), cue_tokens(m.content)) >= 0.10]
            pred = world_model.predict(line) if (enable_world_model and world_model is not None) else None
            decision = adaptive.decide(line, pred, len(evidence))
            if decision.action == "RECALL_EVIDENCE":
                excerpts = "; ".join(f'[{m.timestamp_human}] "{m.content[:180]}"' for m, _ in evidence)
                response = ("RECORDED USER EVIDENCE (not a verified inferred answer): "
                            + excerpts + ". Sonara has not established the answer.")
            elif pred is not None and decision.action == "NEURAL_PREDICTION":
                response = f"NEURAL WORLD PREDICTION: {pred.answer} (confidence={pred.confidence:.2f}, uncertainty={pred.uncertainty:.2f})"
            elif pred is not None:
                response = (f"UNKNOWN: Neural World Model uncertainty is high ({pred.uncertainty:.2f}) or missing observation. "
                            "Use /recall <terms> to inspect recorded evidence.")
            else:
                response = ("UNKNOWN: No trained world-question answering mechanism has been "
                            "validated. Use /recall <terms> to inspect recorded evidence. "
                            "The language-training checkpoint is not a knowledge authority.")

        print("Sonara>", response)
        answer_record = {
            "id": uuid.uuid4().hex[:12], "source": "model", "timestamp": now_iso(),
            "session_id": session_id, "event_index": event_index, "text": response,
            "type": ("neural_prediction" if ask and decision.action == "NEURAL_PREDICTION"
                     else ("evidence_retrieval" if ask and decision.action == "RECALL_EVIDENCE"
                           else ("abstention" if ask else "acknowledgement"))),
            "verification": ("unverified_prediction" if ask and decision.action == "NEURAL_PREDICTION"
                             else ("user_evidence_only" if ask and decision.action == "RECALL_EVIDENCE"
                                   else ("unknown" if ask else "no-claim"))),
            "input_id": record["id"],
        }
        if ask and enable_learned_sparse and world_model is not None and world_model.last_sparse_trace:
            answer_record["sparse_activation"] = dict(world_model.last_sparse_trace)
        if ask and enable_adaptive_loop:
            answer_record["adaptive_gate"] = {
                "action": decision.action, "reason": decision.reason,
                "caution": round(decision.caution, 3),
                "confidence_bar": round(decision.confidence_bar, 3),
                "matching_ratings": decision.supporting_ratings,
            }
        append_record(Path(data_dir) / "events.jsonl", answer_record)
        if ask and enable_adaptive_loop and answer_record["type"] == "neural_prediction":
            last_neural = {"question": line, "response_id": answer_record["id"]}
    append_record(Path(data_dir) / "sessions.jsonl", {"id": session_id, "type": "end", "timestamp": now_iso()})
    print("Session closed. No automatic learning or knowledge promotion occurred.")


# =====================================================================
# 11. WEIGHTS-ONLY EVALUATION — No reads of personal data in this path
# =====================================================================


def show_timeline(data_dir, max_events=30):
    events = [r for r in read_records(Path(data_dir) / "events.jsonl") if r.get("source") == "user"]
    if not events:
        print("No user events yet")
        return
    events.sort(key=lambda r: (r["timestamp"], r["session_id"], r["event_index"]))
    previous = None
    for row in events[-max_events:]:
        present = dt.datetime.fromisoformat(row["timestamp"])
        seconds = (present - previous[0]).total_seconds() if previous and previous[1] == row["session_id"] else None
        elapsed = f" (+{seconds:.0f}s)" if seconds is not None else " (new session)"
        print(f"{row['timestamp']}{elapsed} [{row['session_id']}:{row['event_index']}] {row['text']}")
        previous = (present, row["session_id"])
    print("Order and elapsed time are recorded facts; causal links have NOT been inferred.")


def evaluate_weights_only(args, reader, writer, data, device):
    if not args.question:
        raise ValueError("--question is required for weights-only evaluation")
    # IMPORTANT: Do NOT call read_records / latest_candidates / session files.
    print("STRICT WEIGHTS-ONLY MODE: checkpoint + question only; no session/retrieval access")
    print("  Question:", args.question)
    print("  Neural generation:", answer_with_weights(reader, writer, args.question, data, device))
    if args.expected:
        try:
            with torch.no_grad():
                loss = personal_loss(reader, writer, args.question, args.expected, data, device)
            print("  Supplied-answer teacher-forced token loss:", round(float(loss), 4))
            print("  Lower loss indicates stronger support for THESE GIVEN answer tokens; it does not prove correct free recall.")
        except ValueError as error:
            print("  Cannot score supplied answer:", error)
    print("  Unknown input tokens:", sorted(set(personal_tokens(args.question)) - set(data["word_to_id"])))


# =====================================================================
# 11A. REGRESSION TESTS — Local mechanism checks, no external test modules
# =====================================================================


def self_test():
    torch.manual_seed(5)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        source = {"id": "event1", "session_id": "first", "event_index": 1,
                  "timestamp": "2026-10-13T12:15:00-07:00", "source": "user",
                  "text": "My old dog Jake loved chasing tennis balls."}
        append_record(path / "events.jsonl", source)
        one = create_candidate(path, source, "Who was Jake?", "my old dog")
        two = create_candidate(path, dict(source, id="event2", event_index=2),
                               "What did Jake like?", "tennis balls", split="test")
        reject = create_candidate(path, dict(source, id="event3", event_index=3),
                                  "Should this be trained?", "no")
        append_record(path / "candidates.jsonl", dict(one, status="approved"))
        append_record(path / "candidates.jsonl", dict(two, status="approved"))
        append_record(path / "candidates.jsonl", dict(reject, status="rejected"))
        latest = latest_candidates(path)
        assert [r["event_index"] for r in latest] == [1, 2, 3]
        assert len([r for r in latest if r["status"] == "approved" and r["split"] == "train"]) == 1
        assert len([r for r in latest if r["status"] == "approved" and r["split"] == "test"]) == 1
        assert len([r for r in latest if r["status"] == "rejected"]) == 1
        source_data = encode_data(*build_examples(300, 7)[:2])
        words = len(source_data["words"])
        word = TinyTransformer(words, source_data["max_detail"], 24, 1)
        reader = IdeaReader(words, 24, source_data["label_names"], source_data["word_to_id"][PAD])
        writer = TinyTransformer(words, source_data["max_detail"], 24, 1, conditioned=True)
        baseline_word = word.words.weight.detach().clone()
        added = extend_vocabulary(personal_tokens(one["question"] + " " + one["answer"]), source_data, word, reader, writer)
        assert added >= 1 and "jake" in source_data["word_to_id"]
        assert "tennis" not in source_data["word_to_id"] or "tennis" in list(source_data["words"])[:words]
        assert torch.equal(word.words.weight[:words], baseline_word)
        device = torch.device("cpu")
        reader.eval(); writer.eval()
        old = float(personal_loss(reader, writer, one["question"], one["answer"], source_data, device).detach())
        old_embedding = reader.embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW(list(reader.parameters()) + list(writer.parameters()), lr=0.009)
        for _ in range(15):
            loss = personal_loss(reader, writer, one["question"], one["answer"], source_data, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        new = float(personal_loss(reader, writer, one["question"], one["answer"], source_data, device).detach())
        assert new < old * 0.6, (old, new)
        assert not torch.equal(reader.embedding.weight, old_embedding)
        # The heldout answer and rejected answer have NOT been used in gradients.
        class Args:
            width, layers = 24, 1
        args = Args()
        ckpt = path / "checkpoint.pt"
        save_checkpoint(ckpt, source_data, args, word, reader, writer)
        redata = encode_data(*build_examples(300, 7)[:2])
        wr = TinyTransformer(len(redata["words"]), redata["max_detail"], 24, 1)
        rd = IdeaReader(len(redata["words"]), 24, redata["label_names"], redata["word_to_id"][PAD])
        out = TinyTransformer(len(redata["words"]), redata["max_detail"], 24, 1, conditioned=True)
        load_checkpoint(ckpt, redata, args, wr, rd, out, device)
        restored = float(personal_loss(rd, out, one["question"], one["answer"], redata, device).detach())
        assert abs(restored - new) < 1e-5, (restored, new)
        append_record(path / "learning_events.jsonl", {"candidate_id": one["id"], "fingerprint": candidate_fingerprint(one)})
        assert not has_revoked_or_corrected_training(path)
        append_record(path / "candidates.jsonl", dict(one, status="rejected"))
        assert has_revoked_or_corrected_training(path)
        # Deliberately intercept any storage access while performing inference.
        original_reader = globals()["read_records"]
        def forbidden(*_args, **_kwargs):
            raise AssertionError("weights-only evaluation accessed personal data")
        globals()["read_records"] = forbidden
        try:
            answer_with_weights(rd, out, one["question"], redata, device)
        finally:
            globals()["read_records"] = original_reader
        print("PASS: evidence provenance, event order, review exclusion, vocabulary extension")
        print("PASS: neural gradient, token loss reduction, weight modification")
        print("PASS: checkpoint restoration, correction detection, weights-only file isolation")
        print("NOT PROVED: generalization, true causation, forgetting immunity, self-directed thinking")


# =====================================================================
# 12. RUN — One self-contained file, large minibatches, inspectable results
# =====================================================================

def select_device(value):
    if value != "auto":
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Teach a tiny 3-stage LLM with thousands of paired situations")
    parser.add_argument("--examples", type=int, default=8000, help="Total paired examples including originals (default: 8000)")
    parser.add_argument("--epochs", type=int, default=2, help="Passes for Stages 1 and 3 (default: 2)")
    parser.add_argument("--idea-epochs", type=int, default=8, help="Passes for meaning learner Stage 2 (default: 8)")
    parser.add_argument("--steps", type=int, default=None, help="Optional maximum minibatch updates PER stage (0 = inspect only)")
    parser.add_argument("--batch-size", type=int, default=48, help="Examples processed together (default: 48)")
    parser.add_argument("--layers", type=int, default=2, help="Transformer blocks (default 2)")
    parser.add_argument("--width", type=int, default=64, help="Number of features per word/idea vector (default 64)")
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--ideas-file", default="", help="OPTIONAL editable idea dataset paired with --details-file")
    parser.add_argument("--details-file", default="", help="OPTIONAL extra details file; originals are already embedded")
    parser.add_argument("--idea", default="the cat chases the ball", help="Situation to inspect and expand")
    parser.add_argument("--assess", action="store_true", help="Print per-token prediction alternatives (not an independent reasoning loop)")
    parser.add_argument("--checkpoint", default="LLM_Training_checkpoint_v2.pt", help="Where to save the trained weights")
    parser.add_argument("--resume", action="store_true", help="Load prior weights and keep training, or use with --epochs 0")
    parser.add_argument("--no-save", action="store_true", help="Skip writing a checkpoint")
    parser.add_argument("--adapt", action="store_true", help="OPTIONAL Stage 4 rapid learning comparison")
    parser.add_argument("--adapt-steps", type=int, default=18)
    parser.add_argument("--novel-idea", default="the dog follows the ball")
    parser.add_argument("--novel-detail", default="the dog walks slowly behind the ball .")
    parser.add_argument("--mode", choices=("train", "session", "review", "revise", "learn", "evaluate", "timeline", "self-test", "memories", "experience"), default="train")
    parser.add_argument("--data-dir", default="learning_data", help="Local private session and approval files")
    parser.add_argument("--enable-research", action="store_true",
                        help="Enable explicit /research requests to a laptop-local Ollama model")
    parser.add_argument("--research-model", default="qwen2.5:1.5b-instruct",
                        help="Locally installed Ollama model for /research; no API key or cloud service")
    parser.add_argument("--enable-world-model", action="store_true",
                        help="Enable experimental neural World Model in live session")
    parser.add_argument("--enable-learned-sparse", action="store_true",
                        help="Opt-in research learned sparse neural evidence activation; requires --enable-world-model")
    parser.add_argument("--sparse-budget", type=int, default=8,
                        help="Experimental initial activated event count (1..32)")
    parser.add_argument("--world-checkpoint", default="",
                        help="Path to neural World Model checkpoint")
    parser.add_argument("--enable-adaptive-loop", action="store_true",
                        help="Use cue-scoped explicit feedback to calibrate neural answer caution and bounded recall")
    parser.add_argument("--experience-seeds", type=int, default=3,
                        help="Causal-loop experiment: counterbalanced simulated worlds")
    parser.add_argument("--experience-dim", type=int, default=24,
                        help="Causal-loop experiment: compact world-model width")
    parser.add_argument("--experience-heldout", type=int, default=3,
                        help="Causal-loop experiment: novel actor/object pairs")
    parser.add_argument("--experience-output", default="artifacts/causal_learning_report.json",
                        help="Causal-loop experiment JSON report path")
    parser.add_argument("--candidate-id", default="", help="ID to correct or revoke in --mode revise")
    parser.add_argument("--question", default="", help="Question for strictly weights-only evaluation")
    parser.add_argument("--expected", default="", help="Optional expected answer for teacher-forced score")
    parser.add_argument("--weights-only", action="store_true", help="Explicitly assert no personal records are read")
    parser.add_argument("--personal-steps", type=int, default=70, help="Incremental gradient updates on approved Q/A")
    parser.add_argument("--personal-lr", type=float, default=0.002)
    parser.add_argument("--max-retention-loss-rise", type=float, default=0.35)
    parser.add_argument("--replay-weight", type=float, default=0.5, help="Weight of old general-data replay during personal learning")
    parser.add_argument("--noninteractive", action="store_true", help="List pending reviews without prompting")
    args = parser.parse_args()
    # A packaged starter or personalized checkpoint carries the data/vocabulary
    # layout used to create it, so session/evaluation need not repeat flags.
    if (args.mode in ("session", "learn", "evaluate") or args.resume) and Path(args.checkpoint).exists():
        meta = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        for name, value in meta.get("run_config", {}).items():
            if value is not None and getattr(args, name) == parser.get_default(name):
                setattr(args, name, value)
    if args.epochs < 0 or args.idea_epochs < 0 or (args.steps is not None and args.steps < 0):
        parser.error("--epochs and --steps must not be negative")
    if args.enable_learned_sparse and (not args.enable_world_model or args.mode != "session"):
        parser.error("--enable-learned-sparse requires --mode session --enable-world-model")
    if not 1 <= args.sparse_budget <= 32:
        parser.error("--sparse-budget must be from 1 to 32")
    if args.batch_size < 1 or args.layers < 1 or args.width < 8 or args.adapt_steps < 0 or args.personal_steps < 1 or args.personal_lr <= 0 or args.replay_weight < 0 or args.max_retention_loss_rise < 0:
        parser.error("--batch-size and --layers must be positive; --width at least 8")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))  # Avoid monopolizing a laptop CPU.
    device = select_device(args.device)
    # Local record management does not load the neural model or regenerate data.
    if args.mode == "self-test":
        self_test()
        return
    if args.mode == "experience":
        # Deliberately runs before the language-checkpoint/data pipeline.
        # One existing neural world owner, one existing evidence-memory owner.
        from experiment_causal_loop import run_experiment
        report = run_experiment(
            seeds=args.experience_seeds, dim=args.experience_dim,
            heldout_count=args.experience_heldout, data_dir=args.data_dir,
            output=args.experience_output)
        print(json.dumps(report["summary"], indent=2))
        return
    if args.mode == "review":
        show_review(args.data_dir, interactive=not args.noninteractive)
        return
    if args.mode == "timeline":
        show_timeline(args.data_dir)
        return
    if args.mode == "memories":
        from sonara_memory import SonaraMemorySystem
        mem_sys = SonaraMemorySystem(Path(args.data_dir) / "sonara_memories")
        print("\n=== SONARA'S MEMORY REPOSITORY ===")
        print(f"Total memories stored: {len(mem_sys.memories)}")
        timeline = mem_sys.get_timeline(limit=25)
        if not timeline:
            print("  No memories saved yet. Start a session with --mode session or python3 sonara_memory.py!")
        for item in timeline:
            print(f"  • {item['timestamp_human']} ({item['relative_time']}) [{item['category'].upper()}]")
            print(f"    \"{item['content']}\"")
        journal_path = mem_sys.export_journal()
        print(f"\nMarkdown journal exported to: {journal_path}")
        return
    if args.mode == "revise":
        if not args.candidate_id:
            raise ValueError("--candidate-id is required for --mode revise")
        revise_candidate(args.data_dir, args.candidate_id)
        return
    # Session is a lightweight evidence recorder plus explicit local research.
    # Never initialize synthetic language training or load its weights as a
    # stand-in for grounded world cognition.
    if args.mode == "session":
        run_session(args.data_dir, enable_research=args.enable_research,
                    research_model=args.research_model,
                    enable_world_model=args.enable_world_model,
                    world_checkpoint=args.world_checkpoint,
                    enable_adaptive_loop=args.enable_adaptive_loop,
                    enable_learned_sparse=args.enable_learned_sparse,
                    sparse_budget=args.sparse_budget)
        return
    training, valid, originals_count, synth_count = build_examples(args.examples, args.seed, args.ideas_file, args.details_file)
    data = encode_data(training, valid)
    size = len(data["words"])
    context = data["max_detail"]
    print("\n=== DATA & ARCHITECTURE ===")
    print(f"  Source situations: {originals_count}  | Synthetic variants: {synth_count}")
    print(f"  Train: {len(training)} minibatch rows (includes repeated originals)  | Validation: {len(valid)}  | Vocabulary: {size}")
    print(f"  Batch size: {args.batch_size}  | Epochs: words/detail={args.epochs}, ideas={args.idea_epochs}  | Attention layers: {args.layers}")
    print(f"  Representation width: {args.width}  | Device: {device}")
    print("  NOTICE: Variants share sentence templates; validation does NOT establish deep understanding.")
    print(f"  Example input:  {training[0][0]}")
    print(f"  Example target: {training[0][1]}")

    word_model = TinyTransformer(size, context, args.width, args.layers).to(device)
    reader = IdeaReader(size, args.width, data["label_names"], data["word_to_id"][PAD]).to(device)
    writer = TinyTransformer(size, context, args.width, args.layers, conditioned=True).to(device)
    print(f"  Trainable parameters: {sum(p.numel() for model in (word_model, reader, writer) for p in model.parameters()):,}")
    has_checkpoint = Path(args.checkpoint).exists()
    should_load = args.resume or (args.mode in ("session", "learn", "evaluate") and has_checkpoint)
    if should_load:
        if not has_checkpoint:
            raise FileNotFoundError(f"Checkpoint missing: {args.checkpoint}. Run --mode train first.")
        load_checkpoint(args.checkpoint, data, args, word_model, reader, writer, device)

    if args.mode == "evaluate":
        if not args.weights_only:
            raise ValueError("Evaluation requires --weights-only; personal files are never consulted")
        if not has_checkpoint:
            raise FileNotFoundError("Weights-only evaluation requires a saved checkpoint")
        evaluate_weights_only(args, reader, writer, data, device)
        return
    if args.mode == "learn":
        if not has_checkpoint:
            raise FileNotFoundError("Initialize general model first with --mode train")
        baseline = Path(str(args.checkpoint) + ".general.pt")
        needs_rebuild = has_revoked_or_corrected_training(args.data_dir)
        if needs_rebuild:
            if not baseline.exists():
                raise RuntimeError("A trained fact was revoked/edited, but no original general checkpoint exists. Cannot safely undo the old weights.")
            print("Revoked/corrected training detected: rebuilding personal weights from the preserved general checkpoint")
            # An inverse gradient is not reliable erasure; regenerate the
            # personalized model from the untouched general foundation.
            data = encode_data(training, valid)
            word_model = TinyTransformer(len(data["words"]), data["max_detail"], args.width, args.layers).to(device)
            reader = IdeaReader(len(data["words"]), args.width, data["label_names"], data["word_to_id"][PAD]).to(device)
            writer = TinyTransformer(len(data["words"]), data["max_detail"], args.width, args.layers, conditioned=True).to(device)
            load_checkpoint(baseline, data, args, word_model, reader, writer, device)
            remaining = [r for r in latest_candidates(args.data_dir)
                         if r["status"] == "approved" and r["split"] == "train"]
            if not remaining:
                if args.no_save:
                    print("WARNING: All personal facts revoked but baseline restoration was NOT saved")
                else:
                    save_checkpoint(args.checkpoint, data, args, word_model, reader, writer)
                    print("All approved personal examples revoked; restored general-only checkpoint")
                return
        if learn_approved(args.data_dir, args, data, word_model, reader, writer, device):
            if args.no_save:
                print("WARNING: Neural changes were not saved (--no-save)")
            else:
                if not baseline.exists():
                    shutil.copy2(args.checkpoint, baseline)
                    print(f"Preserved general model checkpoint: {baseline}")
                save_checkpoint(args.checkpoint, data, args, word_model, reader, writer)
        else:
            print("Existing checkpoint unchanged. Review rejected/edited records before using the model.")
        return

    train_stage(1, word_model, reader, data, args, device)
    train_stage(2, reader, reader, data, args, device)
    if not args.resume:
        # Stage 3 starts from learned language weights; only its extra conditioning
        # projection begins randomly. Stage 2 stays frozen while writer learns.
        writer.load_state_dict(word_model.state_dict(), strict=False)
    train_stage(3, writer, reader, data, args, device)

    print("\n=== PREDICTIONS & GENERATION ===")
    print("  Next-word suggestions for 'the cat sees':", next_words(word_model, "the cat sees", data, device))
    vector, meaning = inspect_idea(reader, args.idea, data, device)
    print(f"  Situation: {args.idea}")
    print("  Decoded idea: " + " | ".join(f"{name}={value}" for name, value in meaning.items()))
    print("  Generated detail: " + generate(writer, vector, data, device, trace=args.assess))
    if args.adapt:
        demonstrate_adaptation(writer, reader, data, args, device)
    if not args.no_save:
        save_checkpoint(args.checkpoint, data, args, word_model, reader, writer)


# =====================================================================
# 13. BACKWARDS COMPATIBILITY WRAPPERS FOR EXTERNAL SCRIPTS & EXPERIMENTS
# =====================================================================

def build_data(ideas_file=DEFAULT_IDEAS_FILE, details_file=DEFAULT_DETAILS_FILE, total=320):
    """Build dataset dictionary for the optional language-conditioning ablation."""
    min_needed = len(originals()) + len(extra_events()) + 20
    train, valid, _, _ = build_examples(total=max(min_needed, total), seed=7, external_ideas=ideas_file, external_details=details_file)
    data = encode_data(train, valid)
    data["context_size"] = data["max_detail"]
    data["label_to_id"] = [{name: i for i, name in enumerate(names)} for names in data["label_names"]]
    data["meaning_labels"] = [data["train"]["labels"][:, i] for i in range(len(FIELDS))]
    data["idea_inputs"] = data["train"]["ideas"]
    data["detail_inputs"] = data["train"]["x"]
    data["detail_targets"] = data["train"]["y"]
    data["stream"] = data["train"]["x"].flatten()
    data["examples"] = [
        (ex[0], ex[2][0], ex[2][1], ex[2][2], ex[1])
        for ex in train
    ]
    data["tokens"] = lambda text: [data["word_to_id"].get(w, data["word_to_id"][UNK]) for w in text.lower().split()]
    data["id_to_word"] = {i: w for i, w in enumerate(data["words"])}
    return data


def train_idea_reader(reader, data, steps):
    optimizer = torch.optim.AdamW(reader.parameters(), lr=LEARNING_RATE)
    for _ in range(steps):
        _, guesses = reader(data["idea_inputs"])
        loss = sum(F.cross_entropy(prediction, labels) for prediction, labels in
                   zip(guesses[:len(data["meaning_labels"])], data["meaning_labels"]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item() if steps else None


def train_detail_writer(writer, reader, data, steps):
    reader.eval()
    with torch.no_grad():
        idea_vectors, _ = reader(data["idea_inputs"])
    optimizer = torch.optim.AdamW(writer.parameters(), lr=LEARNING_RATE)
    for _ in range(steps):
        guesses = writer(data["detail_inputs"], idea_vectors)
        loss = F.cross_entropy(
            guesses.reshape(-1, guesses.shape[-1]),
            data["detail_targets"].reshape(-1),
            ignore_index=data["word_to_id"][PAD],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item() if steps else None


def write_detail(writer, idea_vector, data, max_words=36):
    device = next(writer.parameters()).device
    if idea_vector is not None:
        idea_vector = idea_vector.to(device)
    return generate(writer, idea_vector, data, device, trace=False, max_words=max_words)


if __name__ == "__main__":
    main()
