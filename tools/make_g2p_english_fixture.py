#!/usr/bin/env python3
"""English G2P reference fixture for the pure-Pascal misaki port (tasklist
"Kokoro-82M TTS, real checkpoint", Phase 5, task 5.7; tested by 5.1-5.6).

Runs misaki 0.9.4 ONCE on hand-written English sentences, for both accents,
with the G2P built exactly as kokoro's KPipeline builds it for lang_code 'a'
(US) and 'b' (GB), except fallback=None (no espeak; the Pascal port has none):

  en.G2P(trf=False, british=<False|True>, fallback=None, unk='')

Writes, under tests/fixtures/g2p_english/:
  sentences.json   per sentence: group, text, and per accent the G2P result
                   string, every output token (text, tag, whitespace, phonemes,
                   misaki's `_` fields), the tokens left unknown, the phonemes
                   with every heteronym forced to its DEFAULT reading (only when
                   that differs), and KPipeline.en_tokenize's 510-phoneme chunks
                   for the long_mixed group.
  us_gold.json, us_silver.json, gb_gold.json, gb_silver.json
                   SUBSETS of misaki's dictionaries in misaki's own format: every
                   entry whose key (case-insensitive) is a substring of a fixture
                   word, every key misaki looked up while making this fixture
                   (stems, number words, 'minus', 'point', currency units ...),
                   and the single-letter entries. The script re-runs misaki on the
                   subsets and asserts identical output.
  notes.json       versions, the G2P construction, counts per group, unknown
                   token counts, the heteronym DEFAULT-disagreement rate, and the
                   misaki behaviours read from en.py that the port must match.

Coded by Claude (AI).

Usage (from the repo root, under the memory cap):
  systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 \
      /home/bpsa/x/bin/python -I tools/make_g2p_english_fixture.py
"""
import collections
import copy
import json
import os
import re
import sys

import misaki
import num2words
import spacy
from misaki import en
from kokoro.pipeline import KPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "tests", "fixtures", "g2p_english")
DATA = os.path.join(os.path.dirname(os.path.abspath(en.__file__)), "data")
ACCENTS = (("us", False), ("gb", True))
DEFAULT_ONLY_TAG = "\x00DEFAULT"  # a tag no heteronym entry has -> ps['DEFAULT']

GROUPS = [
    ("plain", [
        "The quick brown fox jumps over the lazy dog.",
        "She sells sea shells by the sea shore.",
        "We walked to the market in the morning",
        "This is a simple sentence without anything special",
        "My brother bought a new car last week.",
        "The children played in the garden until dinner.",
        "I think it will rain later today.",
        "Please open the window and let the fresh air in.",
        "An apple a day keeps the doctor away.",
        "They have been waiting for the bus since noon.",
        "Birds fly south when the weather turns cold.",
        "Music makes everything a little bit better.",
        "He asked her where the nearest station was.",
        "Our teacher explained the lesson very clearly.",
        "Every morning I drink a cup of tea.",
        "The river flows through the valley toward the ocean.",
        "Can you hear the thunder in the distance",
        "Hello world",
        "a",
        "I am here, and you are there.",
        "Stand by me, and do not walk by.",
        "Go to the park, then to an office.",
        "The end of the year is near; the evening is calm.",
        "Is it in the box or on it?",
        "Where am I, and who am I?",
    ]),
    ("punctuation", [
        "Wait, what did you say?",
        "Stop! Don't move.",
        "Well... I suppose so.",
        "She paused… then she laughed.",
        "He said: come here; sit down.",
        "It was a long day — a very long day.",
        "It was a long day - a very long day.",
        "It was a long day – a very long day.",
        "It was a long day--a very long day.",
        "\"Hello,\" she said, \"how are you?\"",
        "“Curly quotes” are different from straight ones.",
        "‘Single quotes’ also appear sometimes.",
        "He left (quietly) before the end.",
        "Really?! You did that?",
        "One, two, three; four, five, six.",
        "The list: eggs, milk, bread.",
        "Hmm... well, okay!",
        "Why?",
        "!!!",
        "Is it raining?—No, it is snowing.",
        "[brackets] and {braces} and <angles> too.",
        "The path is /usr/local/bin on this machine.",
        "Email me at someone@example.com today.",
        "Salt & pepper, plus a pinch + more.",
        "Use the * symbol and the # sign.",
        "Visit www.example.org/page for details.",
    ]),
    ("capitals_acronyms", [
        "NASA launched a new rocket.",
        "The FBI and the CIA are agencies.",
        "I work at IBM in the USA.",
        "The BBC reported the news.",
        "Send it via the API using JSON over HTTP.",
        "Our CTO met the CEO at the UN.",
        "The XKCD comic and the QWZ protocol are obscure.",
        "THIS IS ALL CAPS TEXT.",
        "I LOVE YOU.",
        "Hello World From Title Case Words.",
        "The U.S. and the U.K. signed a treaty.",
        "He has a Ph.D. in physics.",
        "Dr. Smith met Mr. Jones and Mrs. Brown.",
        "The iPhone and the eBay app.",
        "We use PyTorch and TensorFlow.",
        "OK, that is fine.",
        "The ZXQ unit failed.",
        "A is the first letter, and B is the second.",
        "My initials are AN, and AM is a morning time.",
        "The IT department fixed the PC.",
        "LOL, that was funny.",
        "She works for NATO and UNESCO.",
        "GPU and CPU speeds differ.",
        "The DNA test was positive.",
        "WiFi and Bluetooth are wireless.",
        "THE END IS NEAR, SAID THE SIGN.",
    ]),
    ("possessives", [
        "The dog's bone is buried.",
        "The dogs' bones are buried.",
        "James's car is red.",
        "The children's toys were everywhere.",
        "My mother's sister's house is far.",
        "Schuler's book was published.",
        "NASA's mission succeeded.",
        "The company's profits rose.",
        "It's the cat's toy, not its own.",
        "The boss's decision surprised everyone.",
        "Charles' idea was good.",
        "The women's team won.",
        "Everyone's opinion matters.",
        "Google's office is big.",
        "The FBI's report came out.",
        "Nguyen's restaurant is popular.",
        "Yesterday's news is old.",
        "The princess's crown sparkled.",
        "Mia's and Leo's bikes are new.",
        "The cat’s whiskers twitched.",
    ]),
    ("contractions", [
        "I'm sure you're right.",
        "They're going, aren't they?",
        "We'll see what he'd say.",
        "She's been here; he's gone.",
        "Don't, won't, can't, shouldn't.",
        "I've got it, haven't I?",
        "You'd better hurry, y'all.",
        "It isn't fair, it wasn't fair.",
        "Let's go, let's go!",
        "Who's there? What's that?",
        "I'd've come if I could've.",
        "They'd known; we'd seen.",
        "Ma'am, it is five o'clock.",
        "Rock 'n' roll never dies.",
        "Y’know, I can’t stay.",
        "Mustn't, needn't, mightn't.",
        "Wouldn't you agree?",
        "That'll do.",
        "There's nothing here.",
        "He ain't coming 'cause it's late.",
    ]),
    ("hyphenated", [
        "A well-known author spoke.",
        "My mother-in-law is visiting.",
        "It was a state-of-the-art system.",
        "The twenty-first century began.",
        "Forty-two is the answer.",
        "A self-driving car stopped.",
        "Long-term plans need short-term wins.",
        "The so-called expert left.",
        "An up-to-date list is ready.",
        "Ice-cream and e-mail.",
        "A one-of-a-kind deal.",
        "Re-enter the code.",
        "The x-ray showed a break.",
        "A sunflower-yellow keyboard-stand.",
        "High-quality, low-cost products.",
        "A nine-year-old girl won.",
        "Pre-trained models help.",
        "The blorp-zing machine hummed.",
        "A T-shirt and a U-turn.",
        "Co-operate with the co-pilot.",
    ]),
    ("suffix_stems", [
        "She googled the answer while googling another.",
        "He microwaved the soup and texted me.",
        "They blogged about blogging.",
        "He vaped outside.",
        "I listen to podcasts on the train.",
        "The post was upvoted and retweeted.",
        "Users were downvoting the upvotes.",
        "We binged the show and livestreamed it.",
        "She is livestreaming the game.",
        "He photoshopped the picture.",
        "They skyped every night; skyping is cheap.",
        "Selfies and hashtags everywhere.",
        "The memes were hashtagged.",
        "Emojis make messages friendlier.",
        "We facetimed grandma.",
        "The smurfs are blue.",
        "He overclocked the chip; overclocking is risky.",
        "The code was refactored; refactoring helps.",
        "She photocopied the downloads.",
        "The text is tokenized before tokenizing again.",
        "The page was hyperlinked and bookmarked.",
        "They cosplayed at the convention; cosplaying is fun.",
        "The vlogs were vlogged in Paris.",
        "The photos were geotagged.",
        "I am bookmarking it.",
        "The film was subtitled.",
        "The records were deduped.",
        "The data was crowdsourced.",
        "He daydreamed and sleepwalked.",
        "The ponies and puppies chased the foxes.",
    ]),
    ("numbers", [
        "I have 3 apples and 12 oranges.",
        "There are 7 days in a week.",
        "She finished 1st, he was 2nd, and I came 3rd.",
        "The 21st century and 42nd Street.",
        "It happened in 1984.",
        "In 2001 and 1066 history changed.",
        "In 2000 and 1900 things were different.",
        "The value of pi is 3.14.",
        "It dropped to -5 degrees.",
        "The population is 1,234,567 people.",
        "A million is 1000000.",
        "He is 101 years old.",
        "Room 404 was empty.",
        "The 1990s were fun.",
        "The score was 3-2.",
        "Version 2.0.1 is out.",
        "It weighs 0.5 kilograms.",
        "Both 0.75 and .5 are fractions.",
        "Chapter 11, page 250.",
        "The 100th visitor and the 1000th.",
        "He ran 26.2 miles.",
        "In the 80s and 90s.",
        "Flight 007 landed.",
        "Between 10 and 20 people came.",
        "The year 2024 is a leap year.",
        "Only 1 left.",
        "He scored 15,000 points.",
        "B2B and P2P networks.",
        "4x4 trucks and 3D movies.",
        "The 12th of May, 1945.",
        "It was 123 or 999 or 1234 or 12345.",
        "Count 0, 13, 19, 40, 77, 100.",
    ]),
    ("currency", [
        "It costs $5.",
        "It costs $5.50.",
        "It costs $1.",
        "It costs $0.99.",
        "The car was $25,000.",
        "A coffee is £3.20 in London.",
        "The ticket costs £1.",
        "He paid €10 for lunch.",
        "That is €2.05 exactly.",
        "I found $1.01 on the floor.",
        "We raised $1,000,000 in total.",
        "The fee is $3.5 or $3.50.",
        "It was 50p or £0.50.",
        "Prices go from $5 to $10.",
        "She earned $100 and spent $45.30.",
        "The stock is at $123.456.",
        "A pound is £1 and a euro is €1.",
        "They owe us $12 million.",
        "That is ¥500 in yen.",
        "Only $.99 today.",
    ]),
    ("percent", [
        "The price rose by 5%.",
        "About 50% of people agree.",
        "It is 100% true.",
        "Growth was 3.5% last year.",
        "Rates fell 0.25%.",
        "Battery at 12 percent.",
        "A 20% discount and a 15 % tip.",
        "Up 1,000% in a decade.",
        "The odds are 99.9%.",
        "Fat content changed by -2%.",
    ]),
    ("heteronyms", [
        "I read books every day.",
        "Yesterday I read a book.",
        "Please read the record before you record a new one.",
        "They live in a city with live music.",
        "She will present the present to him.",
        "I object to that object.",
        "The wind will wind down soon.",
        "A tear rolled down; do not tear the page.",
        "Close the door; stay close to me.",
        "What is the use? We use it daily.",
        "Lead the team; lead is heavy.",
        "The farm will produce fresh produce.",
        "Do not desert me in the desert.",
        "Wait a minute; the change is minute.",
        "You need a permit; we permit it.",
        "They refuse to collect the refuse.",
        "I am content with the content.",
        "The wound was wound with a bandage.",
        "Take a bow; the bow broke.",
        "Row the boat in a row.",
        "He plays bass; he caught a bass.",
        "Does he know the does?",
        "Conduct the orchestra with good conduct.",
        "Sign the contract; muscles contract.",
        "The project will project an image.",
        "Change the subject; subject them to tests.",
        "Separate the separate piles.",
        "Estimate the cost; the estimate is high.",
        "We increase the increase.",
        "The rebel will rebel.",
        "Polish the Polish silver.",
        "Excuse me, that is no excuse.",
        "They house people in a house.",
        "Have you read it? I have read it.",
        "He reads; she read; they are reading.",
        "We recorded the records.",
        "I used to use this; it is used.",
        "Close it! It is close.",
    ]),
    ("unknown_and_names", [
        "We walked through the sunflowerfield.",
        "Put it on the keyboardstand.",
        "The coffeecup is empty.",
        "A rainbowbridge appeared.",
        "The moonlightgarden glowed.",
        "Joao Paulo Schuler wrote this.",
        "Nguyen and Schuler met Joao.",
        "Professor Xochitl and Ms. Siobhan Okonkwo.",
        "The blorptastic zingwhizzle.",
        "Visit Llanfairpwllgwyngyll in Wales.",
        "Zbigniew Brzezinski spoke.",
        "Dmitri Shostakovich composed it.",
        "The village of Qwrtz is small.",
        "Szczecin and Wrocław are cities.",
        "Café, naïve, résumé.",
        "Mr. Okafor and Ms. Tanaka.",
        "Hrrm, pfft, shh.",
        "The firetruckladder was long.",
        "Greenhousegas levels rise.",
        "The snowmanhat fell off.",
        "Bpsa and joaopauloschuler.",
        "The Kokoro voice says hello.",
        "Raphaël and Zoë visited.",
        "Supercalifragilisticexpialidocious!",
    ]),
    ("long_mixed", [
        "On March 3rd, 1984, Dr. Smith's team at NASA spent $2,500,000 on a "
        "state-of-the-art telescope; by 2001 they'd recorded 1,234 new stars, "
        "and the FBI's report said 95% of them were faint. \"We can't stop "
        "now,\" she said — and they didn't. The sunflowerfield behind the "
        "lab, which Joao Schuler photographed every morning, grew twelve feet "
        "tall. Why? Nobody knew... but the children's teachers loved it, and "
        "the well-known reporters from the BBC googled it for hours.",
        "When I read the record last year, I couldn't believe that the lead "
        "singer would live in a desert town and refuse to present the present "
        "he'd bought for £3.20. The wind was strong, the minute hand was "
        "slow, and the tear in his coat let the cold in. He'd object to every "
        "object, use every excuse, and close every door; still, his records "
        "sold 15,000 copies in the 1990s, about 12% of the market, which is "
        "more than most people's first albums ever do.",
        "Our CTO said the API was up-to-date, but the JSON parser crashed on "
        "version 2.0.1 at 3:45 in the morning. We'd refactored the tokenizer, "
        "deduped the records, and livestreamed the fix to 101 users, yet the "
        "bug came back: a self-driving test car paid €10.50 for parking, "
        "twice! It's the 21st century, she sighed; machines shouldn't make "
        "mistakes like that. The vlogs we posted got 1,000 upvotes, the memes "
        "got hashtagged, and somebody's grandmother facetimed us to ask what "
        "a keyboardstand was.",
        "Once upon a time, in a village called Qwrtz, there lived an old "
        "woman, her nine-year-old grandson, and a cat named Mr. Whiskers. "
        "Every evening the boy would ask, \"Grandma, why is the sky dark?\" "
        "and every evening she would answer, \"Because the sun is tired, "
        "dear.\" One day a stranger arrived with -5 dollars, 3 apples, and a "
        "map of the moonlightgarden; he said he'd walked 26.2 miles since "
        "dawn. The villagers didn't trust him at first, but by the 100th day "
        "they called him family. Years later, the grandson, now a grown man "
        "with children of his own, told the story at every wedding, every "
        "birthday, and every funeral, and nobody ever got tired of hearing it.",
        "The quarterly report shows revenue of $45.30 per user, up 3.5% from "
        "the previous quarter, while costs fell 0.25%. Our engineers in the "
        "USA, the U.K., and Vietnam shipped 7 releases; the IT team replaced "
        "404 laptops. Nguyen's group overclocked the GPU cluster, but "
        "overclocking is risky, so they're testing it again. Meanwhile, the "
        "marketing team's well-designed campaign, which blogged about blogging "
        "and podcasts, reached 1,234,567 people in 42 countries — "
        "a company record.",
        "I'm not sure whether you've heard, but the old lighthouse on the "
        "northern coast, the one built in 1066 according to legend, is going "
        "to be turned into a museum. They'll repair the stairs, polish the "
        "brass, and present the history of the coast in twelve rooms. Tickets "
        "will cost £1 for children and £5.50 for adults; it's a "
        "bargain, honestly. If the weather holds, they'll open on the 1st of "
        "June, and the mayor's going to cut the ribbon herself, wearing the "
        "town's ancient hat.",
    ]),
]

BEHAVIOURS = {
    "unknown_word": (
        "With fallback=None, Lexicon.__call__ returns (None, None) for a word it "
        "cannot resolve: token.phonemes stays None (null in sentences.json), "
        "the token keeps its text, tag and whitespace, and G2P.__call__ writes "
        "unk ('') in the result string, so the word simply disappears from the "
        "phoneme string (the space after it stays). KPipeline.en_tokenize then "
        "turns None into ''. Inside a multi-piece word (pieces from "
        "subtokenize with no whitespace between them, e.g. a hyphenated word), "
        "the loop at en.py:668 tries the longest merge of pieces ending at the "
        "right first; an unresolvable piece stays None and merge_tokens(unk='') "
        "joins it as '', so known pieces keep their phonemes and unknown ones "
        "vanish. Words with characters outside A-Z a-z ' - (e.g. café, "
        "Wrocław) are unknown unless the exact key is in the dictionary. "
        "Pieces made only of ',-._‘’/ get ''."),
    "capitals_and_acronyms": (
        "Stress offset from case (Lexicon.__call__): lowercase None, "
        "Capitalized 0.5, ALL CAPS 2, applied with apply_stress. get_word "
        "lowercases a capitalized/ALL-CAPS word that is not itself a key when "
        "the lowercase form (or its -s/-ed/-ing stem) is known, except "
        "NNP-tagged words of 7 letters or less. is_known treats any word "
        "whose letters after the first are all upper case as known; lookup "
        "then lowercases an ALL-CAPS word that is not a gold key, and when "
        "nothing is found (or the word is NNP and its entry has no primary "
        "stress) get_NNP spells it with the gold single-letter entries, "
        "demoting all stresses to secondary and making the LAST one primary. "
        "Dotted abbreviations (U.S., Ph.D.) whose parts are at most 2 letters "
        "are also spelled by get_NNP. So the 'spell unknown ALL-CAPS' route in "
        "plan 5.5 is misaki's own rule, not a fallback."),
    "special_cases": (
        "get_special_case: 'a' -> ɐ when tagged DT else ˈA; "
        "am/an/I/by/to/in/the/used have context forms (the -> ði before "
        "a vowel; to -> tə / tʊ by next vowel; I tagged PRP -> "
        "ˌI; 'by' tagged as adverb -> bˈI; 'used' + following 'to' "
        "-> the VBD reading); 'vs' tagged IN -> versus; symbols % & + @ -> "
        "percent/and/plus/at; '.' and '/' are read 'dot'/'slash' only when "
        "spaCy tags the token ADD (URLs/e-mail). These need the part-of-speech "
        "tag (DT, PRP, IN, ADV, VBD, NNP, ADD); the Pascal port has no tagger."),
    "context": (
        "Tokens are processed right to left; TokenContext carries "
        "future_vowel (does the next word start with a vowel, None after "
        "punctuation) and future_to. A heteronym entry with a 'None' key is "
        "used when future_vowel is None, i.e. at the end of a clause; this is "
        "not a spaCy decision and the port can reproduce it."),
    "suffix_rules": (
        "Only when the whole word is not known, one level deep, stem must "
        "satisfy is_known: stem_s: -s (not -ss), -'s, -es (len>4, not -ies), "
        "-ies -> y (len>4); appends s after p t k f θ, ᵻz (US) / "
        "ɪz (GB) after s z ʃ ʒ ʧ ʤ, else z. stem_ed: "
        "-d (not -dd), -ed (len>4, not -eed); appends t after p k f θ "
        "ʃ s ʧ, ᵻd/ɪd after d, d after other non-t, and "
        "after t: GB ɪd, US ɾᵻd when the sound before t is in "
        "AIOWYiuæɑəɛɪɹʊʌ, else "
        "ᵻd. stem_ing (len>=5): stem, stem+e, or doubled consonant / "
        "-cking (drop one letter); appends ɪŋ, US t after such a "
        "vowel -> ɾɪŋ, GB returns None when the stem ends in "
        "ə or ː; stem_ing is called with stress 0.5 when the word "
        "had none. Also: word ending s' looks up word's, a trailing ' is "
        "dropped. The same _s/_ed/_ing endings apply to numbers (1990s)."),
    "numbers": (
        "is_number: digits with , and . (and a leading - only at the head of a "
        "word), optional suffix ing 'd ed 's st nd rd th s. get_number: "
        "ordinal suffix -> num2words(to='ordinal'); 4-digit integer not after "
        "a currency sign -> num2words(to='year'); digits glued to letters "
        "(not head, e.g. 'B2B', '3D') -> read as digits/short groups; numbers "
        "with 2+ dots (2.0.1) -> each group, groups with a leading 0 or not "
        "of length 2 read digit by digit; $ £ € followed by a "
        "number with at most 2 decimals -> dollars/cents, pounds/pence, "
        "euros/cents ('and' between, singular for 1, pence never pluralized, "
        "zero part dropped); otherwise cardinal via num2words (float for "
        "decimals, '.5' -> point five). Commas are removed. num2words output "
        "is split on non-letters and each word looked up; the word 'and' is "
        "DROPPED ('one hundred one'), 'point' gets stress -2 (unstressed), "
        "'minus' prefixes a leading '-'. A currency sign whose number is "
        "followed by another CD token ('$12 million') gets the plural unit "
        "appended after the number words. '¥' is not a currency. num_flags "
        "(& keep 'and', a 'a' for one, n) only come from the [n](#flags#) "
        "markdown syntax."),
    "stress": (
        "Primary ˈ and secondary ˌ are written before the vowel of "
        "the stressed syllable (restress moves an added mark to just before "
        "the first vowel after it). apply_stress(ps, s): s<-1 removes all "
        "marks; s==-1, or s in (0,-0.5) with a primary present, demotes "
        "primary to secondary; s in (0,0.5,1) with no marks adds secondary; "
        "s>=1 with only secondary promotes it; s>1 with no marks adds "
        "primary. Inside a multi-piece word, resolve_tokens demotes "
        "(-0.5) the lighter half of the pieces that carry a primary "
        "(stress_weight counts diphthong letters AIOQWYʤʧ double), "
        "unless the pieces are separated by space, '/', or mix letters and "
        "digits."),
    "post_processing": (
        "G2P version None (KPipeline's default) replaces ɾ with T and "
        "ʔ with t in every token after lookup (en.py:707), so US flaps "
        "come out as T. The dictionaries themselves contain ɾ."),
    "punctuation": (
        "Tokens tagged as punctuation by spaCy (. , -LRB- -RRB- `` '' : $ # "
        "NFP ...) keep only the characters in ;:,.!?—…\"“”; "
        "( ) map to ( ), opening quotes to “, closing to ”; a '-' or "
        "'–' tagged ':' becomes —; other symbols (*, #, [, {, <) give "
        "''. Punctuation depends on the spaCy tag, not only on the character."),
    "other_input_syntax": (
        "G2P.preprocess honours markdown links: [word](/IPA/) sets the "
        "phonemes, [word](-1|-0.5|0.5|+1|2...) sets stress, [n](#flags#) "
        "sets number flags. A '2' between letters (P2P) is read 'to' "
        "(alias). The Pascal port may skip this syntax but should not "
        "misread brackets."),
    "chunking": (
        "KPipeline.__call__ first splits the text on \\n+, then "
        "en_tokenize cuts the token list so each chunk's phoneme string is "
        "at most 510 characters, backing up to the last token whose "
        "phonemes are one of !.?…, then :;, then ,— (plus a "
        "following ) or ”), else cutting at the current token; a chunk "
        "longer than 510 is truncated with a warning."),
}

OBSERVED = {
    "suffix_gaps": "stem_ed does not undo a doubled consonant (blogged, vlogged, "
                   "hashtagged, photoshopped, geotagged; GB also hummed) nor "
                   "-ied -> y (photocopied), and -ed/-ing only try the stem misaki has: "
                   "livestreamed, facetimed, deduped, smurfs stay unknown because "
                   "livestream, facetime, dedupe, smurf are not keys. GB 'refactoring' "
                   "is unknown because GB _ing refuses a stem ending in \u0259.",
    "capitalized_unknown_names": "Capitalized unknown words (Joao, Schuler, Nguyen, "
                                 "Okafor) are NOT spelled: is_known only accepts words "
                                 "whose letters after the first are upper case, so they "
                                 "come out empty. Only ALL-CAPS strings are spelled.",
    "camel_case": "subtokenize splits camelCase; unknown pieces vanish: 'WiFi' -> '', "
                  "'PyTorch' -> 't\u02c8\u0254\u0279\u02a7' (Py dropped).",
    "spacy_errors": "spaCy's tag is sometimes wrong: 'I read books every day' -> "
                    "\u0279\u02c8\u025bd, 'Have you read it?' -> \u0279\u02c8id, "
                    "'A is the first letter' -> \u0250 (tagged DT). DEFAULT is not "
                    "always worse than misaki.",
    "pos_keyed_function_words": "'that' (DT vs DEFAULT, stressed vs not) and 'IT' are "
                                "POS-keyed too; 'that' is the most frequent disagreement.",
    "lead_not_heteronym": "'lead' is a plain string (l\u02c8id) in both accents.",
    "dropped_punctuation": "' - ' with spaces (spaCy HYPH) and '--' become '' (no dash), "
                           "and '--' between words leaves no space; '?\u2014' inside "
                           "'raining?\u2014No' is lost (one spaCy token); [ ] and < > "
                           "vanish, { } become ( ); * # give ''.",
    "four_digit_numbers": "Every bare 4-digit integer is read as a year (1234 -> twelve "
                          "thirty four); leading zeros vanish (007 -> seven).",
    "currency_details": "\u00a5 is not a currency (\u00a5500 -> five hundred); "
                        "$123.456 (3 decimals) is read as a plain decimal without "
                        "'dollars'; '$12 million' -> twelve million dollars.",
    "urls": "'/usr/local/bin' loses 'usr'; e-mail and URL dots/slashes become "
            "dot/at only when spaCy tags them ADD.",
}


class RecordingDict(dict):
    """dict that remembers every key asked for (get, in, [])."""

    def __init__(self, base, seen):
        super().__init__(base)
        self.seen = seen

    def get(self, key, default=None):
        self.seen.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.seen.add(key)
        return super().__contains__(key)

    def __getitem__(self, key):
        self.seen.add(key)
        return super().__getitem__(key)


def build_g2p(british):
    # kokoro/pipeline.py line 113, with fallback=None.
    return en.G2P(trf=False, british=british, fallback=None, unk='')


def force_default_heteronyms(g2p):
    """Patches g2p.lexicon.lookup so every POS-keyed entry resolves to DEFAULT."""
    lexicon = g2p.lexicon
    original_lookup = lexicon.lookup

    def lookup(word, tag, stress, ctx):
        probe = word
        if probe == probe.upper() and probe not in lexicon.golds:
            probe = probe.lower()
        if isinstance(lexicon.golds.get(probe), dict):
            tag = DEFAULT_ONLY_TAG
        return original_lookup(word, tag, stress, ctx)

    lexicon.lookup = lookup


UNDERSCORE_DEFAULTS = {("is_head", True), ("prespace", False), ("num_flags", "")}


def dump_token(tk):
    underscore = {k: v for k, v in dict(tk._).items()
                  if v is not None and (k, v) not in UNDERSCORE_DEFAULTS}
    token = {"text": tk.text, "tag": tk.tag, "whitespace": tk.whitespace,
             "phonemes": tk.phonemes, "_": underscore}
    if getattr(tk, "rating", None) is not None:  # en.py:663 sets w.rating, not w._.rating
        token["rating"] = tk.rating
    return token


def run_accent(g2p, sentences):
    results = []
    for group, text in sentences:
        phonemes, tokens = g2p(text)
        results.append((phonemes, tokens))
    return results


def write_one_token_per_line(path, document):
    """Indented JSON, except that each token object sits on one line."""
    compact = []

    def placeholder(token):
        compact.append(json.dumps(token, ensure_ascii=False))
        return f"\x00TOKEN{len(compact) - 1}\x00"

    for rec in document["sentences"]:
        for accent, _ in ACCENTS:
            rec[accent]["tokens"] = [placeholder(t) for t in rec[accent]["tokens"]]
    text = json.dumps(document, ensure_ascii=False, indent=1)
    text = re.sub(r'"\\u0000TOKEN(\d+)\\u0000"', lambda m: compact[int(m.group(1))], text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    assert json.load(open(path, encoding="utf-8"))["sentences"][0]["us"]["tokens"][0]["text"]


def substrings(words):
    out = set()
    for w in words:
        for i in range(len(w)):
            for j in range(i + 1, len(w) + 1):
                out.add(w[i:j])
    return out


def main():
    sentences = [(group, text) for group, texts in GROUPS for text in texts]
    os.makedirs(FIX, exist_ok=True)
    records = [{"id": i, "group": g, "text": t} for i, (g, t) in enumerate(sentences)]
    notes_counts = {}
    subset_files = {}
    hetero_stats = {}
    for accent, british in ACCENTS:
        g2p = build_g2p(british)
        seen = set()
        full_golds, full_silvers = g2p.lexicon.golds, g2p.lexicon.silvers
        g2p.lexicon.golds = RecordingDict(full_golds, seen)
        g2p.lexicon.silvers = RecordingDict(full_silvers, seen)
        main_run = run_accent(g2p, sentences)
        default_g2p = build_g2p(british)
        default_g2p.lexicon.golds = RecordingDict(default_g2p.lexicon.golds, seen)
        default_g2p.lexicon.silvers = RecordingDict(default_g2p.lexicon.silvers, seen)
        force_default_heteronyms(default_g2p)
        default_run = run_accent(default_g2p, sentences)

        per_group = collections.OrderedDict((g, {"sentences": 0, "tokens": 0, "unknown_tokens": 0,
                                                 "heteronym_tokens": 0, "heteronym_tokens_differing": 0,
                                                 "tokens_differing_from_default_run": 0})
                                            for g, _ in GROUPS)
        differing_words = collections.Counter()
        for rec, (phonemes, tokens), (default_phonemes, default_tokens) in zip(records, main_run, default_run):
            stats = per_group[rec["group"]]
            stats["sentences"] += 1
            stats["tokens"] += len(tokens)
            unknown = [tk.text for tk in tokens if tk.phonemes is None]
            stats["unknown_tokens"] += len(unknown)
            assert [tk.text for tk in tokens] == [tk.text for tk in default_tokens]
            for tk, dtk in zip(tokens, default_tokens):
                if tk.phonemes != dtk.phonemes:  # includes inflected forms (lived) via stems
                    stats["tokens_differing_from_default_run"] += 1
                letters = re.sub(r"[^A-Za-z]", "", tk.text)
                if isinstance(full_golds.get(letters), dict) or isinstance(full_golds.get(letters.lower()), dict):
                    stats["heteronym_tokens"] += 1
                    if tk.phonemes != dtk.phonemes:
                        stats["heteronym_tokens_differing"] += 1
                        differing_words[tk.text.lower()] += 1
            entry = {"phonemes": phonemes, "unknown_tokens": unknown,
                     "tokens": [dump_token(tk) for tk in tokens]}
            if default_phonemes != phonemes:
                entry["phonemes_if_default"] = default_phonemes
            if rec["group"] == "long_mixed":
                chunks = [{"text": gs, "phonemes": ps}
                          for gs, ps, _ in KPipeline.en_tokenize(None, copy.deepcopy(tokens))]
                assert len(phonemes.strip()) > 510 and len(chunks) > 1, (len(phonemes.strip()), rec["text"][:40])
                entry["chunks"] = chunks
            rec[accent] = entry
        total = {k: sum(s[k] for s in per_group.values()) for k in next(iter(per_group.values()))}
        notes_counts[accent] = {"per_group": per_group, "total": total}
        hetero_stats[accent] = {
            "rule": "a token counts when its letters, lowercased, are a POS-keyed gold "
                    "entry; it differs when misaki's output token differs from a run "
                    "with every POS-keyed entry forced to DEFAULT (same spaCy tokens, "
                    "same right-to-left context)",
            "all_sentences": {"heteronym_tokens": total["heteronym_tokens"],
                              "differing": total["heteronym_tokens_differing"],
                              "any_token_differing": total["tokens_differing_from_default_run"]},
            "heteronyms_group": {"heteronym_tokens": per_group["heteronyms"]["heteronym_tokens"],
                                 "differing": per_group["heteronyms"]["heteronym_tokens_differing"]},
            "differing_words": dict(differing_words.most_common()),
        }

        words = set()
        for _, text in sentences:
            for chunk in text.lower().split():
                words.add(chunk)
                words.update(re.findall(r"[a-z']+", chunk))
        wanted = substrings(words) | {k.lower() for k in seen if isinstance(k, str)}
        wanted |= set("abcdefghijklmnopqrstuvwxyz")
        for kind in ("gold", "silver"):
            name = f"{accent}_{kind}.json"
            with open(os.path.join(DATA, name), encoding="utf-8") as f:
                full = json.load(f)
            subset = {k: v for k, v in full.items() if k.lower() in wanted}
            path = os.path.join(FIX, name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(subset, f, ensure_ascii=False, indent=2)
                f.write("\n")
            if kind == "gold":
                hetero_stats[accent]["pos_keyed_entries_in_gold_file"] = sum(
                    isinstance(v, dict) for v in full.values())
            subset_files[name] = {"entries": len(subset), "full_entries": len(full),
                                  "bytes": os.path.getsize(path)}

        check = build_g2p(british)
        for kind, attr in (("gold", "golds"), ("silver", "silvers")):
            with open(os.path.join(FIX, f"{accent}_{kind}.json"), encoding="utf-8") as f:
                setattr(check.lexicon, attr, en.Lexicon.grow_dictionary(json.load(f)))
        for (phonemes, tokens), (sub_phonemes, sub_tokens) in zip(main_run, run_accent(check, sentences)):
            assert phonemes == sub_phonemes, (phonemes, sub_phonemes)
            assert [tk.phonemes for tk in tokens] == [tk.phonemes for tk in sub_tokens]
        print(f"{accent}: subset dictionaries reproduce misaki's output exactly")

    write_one_token_per_line(os.path.join(FIX, "sentences.json"),
                             {"misaki_version": misaki.__version__, "sentences": records})
    notes = {
        "generator": "tools/make_g2p_english_fixture.py",
        "versions": {"misaki": misaki.__version__, "spacy": spacy.__version__,
                     "spacy_model": "en_core_web_sm " + spacy.load("en_core_web_sm").meta["version"],
                     "num2words": getattr(num2words, "__version__", "0.5.14")},
        "g2p_construction": {
            "us": "misaki.en.G2P(trf=False, british=False, fallback=None, unk='')",
            "gb": "misaki.en.G2P(trf=False, british=True, fallback=None, unk='')",
            "source": "kokoro/pipeline.py builds en.G2P(trf=trf, british=lang_code=='b', "
                      "fallback=EspeakFallback or None, unk=''); trf defaults to False; "
                      "version stays None"},
        "sentence_fields": {
            "phonemes": "first value returned by G2P.__call__ (tokens joined with their "
                        "whitespace, unknown tokens as '')",
            "tokens": "second value: the merged output tokens; phonemes null = unknown",
            "phonemes_if_default": "present only when forcing DEFAULT for every POS-keyed "
                                   "entry changes the result (what task 5.4 will produce)",
            "chunks": "long_mixed only: KPipeline.en_tokenize output (text, phonemes)"},
        "dictionary_subsets": {
            "rule": "entry kept when key.lower() is a substring of a lowercased fixture "
                    "word (whitespace chunk or letter/apostrophe run), or was looked up "
                    "by misaki while producing this fixture (either accent run, incl. "
                    "the DEFAULT run), or is a single letter",
            "files": subset_files,
            "total_bytes": sum(v["bytes"] for v in subset_files.values()),
            "verified": "misaki re-run with the subsets gives identical output"},
        "counts": notes_counts,
        "heteronyms": hetero_stats,
        "behaviours_from_en_py": BEHAVIOURS,
        "observed_in_this_fixture": OBSERVED,
    }
    with open(os.path.join(FIX, "notes.json"), "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {len(records)} sentences, subsets {notes['dictionary_subsets']['total_bytes']} bytes")
    for accent in notes_counts:
        print(accent, notes_counts[accent]["total"], hetero_stats[accent]["all_sentences"],
              hetero_stats[accent]["heteronyms_group"])


if __name__ == "__main__":
    sys.exit(main())
