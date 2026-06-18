# Supported Languages for PII Masking

This document tracks the status of Stanza to ONNX conversion for 52 target languages.

| # | Language | Code | Tokenizer | POS | Depparse | Status | Notes |
| --- | ---------- | ------ | [ ] | [ ] | [ ] | -------- | ------- |
| 1 | Albanian | sq | [x] | [x] | [x] | ❌ Failed | Pkg: combined_nocharlm, CharLM |
| 2 | Arabic | ar | [x] | [x] | [x] | ❌ Failed | Pkg: default |
| 3 | Azerbaijani | az | [ ] | [ ] | [ ] | ❌ Failed | Pkg: default |
| 4 | Bengali | bn | [ ] | [ ] | [ ] | ❌ Failed | Pkg: default |
| 5 | Bulgarian | bg | [x] | [x] | [x] | ❌ Failed | Pkg: default |
| 6 | Catalan | ca | [x] | [x] | [x] | ❌ Failed | Pkg: default, CharLM |
| 7 | Chinese | zh-hans | [x] | [x] | [ ] | ⚠️ Partial |  |
| 8 | Croatian | hr | [x] | [x] | [x] | ❌ Failed | Pkg: default, CharLM |
| 9 | Czech | cs | [x] | [x] | [x] | ❌ Failed | Pkg: default, CharLM |
| 10 | Danish | da | [x] | [x] | [x] | ❌ Failed | Pkg: default |
| 11 | Dutch | nl | [x] | [x] | [x] | ❌ Failed | Pkg: default |
| 12 | English | en | [x] | [x] | [x] | ❌ Failed | Pkg: combined_nocharlm, CharLM |
| 13 | Estonian | et | [x] | [x] | [x] | ❌ Failed | Pkg: default, CharLM |
| 14 | Finnish | fi | [x] | [x] | [x] | ❌ Failed | Pkg: combined_nocharlm, CharLM |
| 15 | French | fr | [x] | [x] | [x] | ❌ Failed | Pkg: combined_nocharlm, CharLM |
| 16 | Georgian | ka | [x] | [x] | [x] | ❌ Failed | Pkg: default, CharLM |
| 17 | German | de | [x] | [x] | [x] | ❌ Failed | Pkg: combined_nocharlm, CharLM |
| 18 | Greek | el | [x] | [x] | [x] | ❌ Failed | Pkg: default, CharLM |
| 19 | Hebrew | he | [x] | [x] | [x] | ❌ Failed | Pkg: combined_nocharlm, CharLM |
| 20 | Hindi | hi | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 21 | Hungarian | hu | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?), CharLM |
| 22 | Icelandic | is | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?), CharLM |
| 23 | Indonesian | id | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 24 | Italian | it | [x] | [x] | [x] | ❌ Failed | Pkg: combined_nocharlm, CharLM |
| 25 | Japanese | ja | [x] | [x] | [x] | ❌ Failed | Pkg: combined_nocharlm, CharLM |
| 26 | Kazakh | kk | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 27 | Khmer | km | [ ] | [ ] | [ ] | ❌ Failed | Pkg: standard (charlm?) |
| 28 | Korean | ko | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?), CharLM |
| 29 | Latvian | lv | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 30 | Lithuanian | lt | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 31 | Malay | ms | [ ] | [ ] | [ ] | ❌ Failed | Pkg: standard (charlm?) |
| 32 | Marathi | mr | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 33 | Norwegian | no | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 34 | Persian | fa | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 35 | Polish | pl | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 36 | Portuguese | pt | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 37 | Romanian | ro | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?), CharLM |
| 38 | Russian | ru | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 39 | Serbian | sr | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?), CharLM |
| 40 | Slovak | sk | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?), CharLM |
| 41 | Slovenian | sl | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 42 | Spanish | es | [x] | [x] | [x] | ❌ Failed | Pkg: combined_nocharlm |
| 43 | Swahili | sw | [ ] | [ ] | [ ] | ❌ Failed | Pkg: standard (charlm?) |
| 44 | Swedish | sv | [x] | [x] | [x] | ❌ Failed | Pkg: standard (charlm?) |
| 45 | Tagalog | tl | [ ] | [ ] | [ ] | ❌ Failed | Pkg: standard (charlm?) |
| 46 | Telugu | te | [x] | [x] | [ ] | ❌ Failed | Pkg: standard (charlm?) |
| 47 | Thai | th | [x] | [ ] | [ ] | ❌ Failed | Pkg: standard (charlm?) |
| 48 | Turkish | tr | [x] | [ ] | [ ] | ❌ Failed | Pkg: standard (charlm?) |
| 49 | Ukrainian | uk | [x] | [ ] | [ ] | ❌ Failed | Pkg: standard (charlm?) |
| 50 | Urdu | ur | [ ] | [ ] | [ ] | Pending | |
| 51 | Uzbek | uz | [ ] | [ ] | [ ] | Pending | |
| 52 | Vietnamese | vi | [ ] | [ ] | [ ] | Pending | |
