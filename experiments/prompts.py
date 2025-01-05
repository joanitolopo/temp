prompt_instruction = [
    # Variasi Formal dan Langsung
    'Terjemahkan teks berikut dari bahasa {SOURCE} ke bahasa {TARGET}.\nTeks: {INPUT}\nTerjemahan:',
    'Teks dalam bahasa {SOURCE}: {INPUT}\nTerjemahkan ke dalam bahasa {TARGET}:',
    'Berikan terjemahan teks berikut dari bahasa {SOURCE} ke bahasa {TARGET}.\nTeks: {INPUT}',
    'Silakan terjemahkan teks berikut dari bahasa {SOURCE} ke bahasa {TARGET}:\n{INPUT}',
    'Teks berikut dalam bahasa {SOURCE}: {INPUT}\nTolong terjemahkan ke bahasa {TARGET}.',
    
    # Variasi Pertanyaan Langsung
    'Apa terjemahan dari teks ini dalam bahasa {TARGET}?\nTeks: {INPUT}',
    '{INPUT}\nApa artinya dalam bahasa {TARGET}?',
    'Bagaimana teks berikut diterjemahkan ke bahasa {TARGET}?\nTeks: {INPUT}',
    'Teks: {INPUT}\nApa hasil terjemahannya dalam bahasa {TARGET}?',
    'Jika teks dalam bahasa {SOURCE} adalah {INPUT}, apa artinya dalam bahasa {TARGET}?',
    
    # Variasi Berbasis Konteks
    'Berikut adalah teks dalam bahasa {SOURCE}: {INPUT}\nTolong terjemahkan ke bahasa {TARGET}.',
    'Diberikan teks berikut dalam bahasa {SOURCE}: {INPUT}\nMohon terjemahkan ke bahasa {TARGET}.',
    'Diberikan kalimat dalam bahasa {SOURCE}: {INPUT}\nApa artinya dalam bahasa {TARGET}?',
    '{INPUT} adalah teks dalam bahasa {SOURCE}.\nSilakan terjemahkan ke bahasa {TARGET}.',
    'Teks berikut ditulis dalam bahasa {SOURCE}: {INPUT}\nBerikan terjemahannya dalam bahasa {TARGET}.',
    
    # Variasi Instruksi Spesifik
    'Anda diminta untuk menerjemahkan teks berikut dari bahasa {SOURCE} ke bahasa {TARGET}.\nTeks: {INPUT}',
    'Terjemahkan teks berikut dari bahasa {SOURCE} ke bahasa {TARGET}, sesuai dengan konteks.\nTeks: {INPUT}',
    'Teks berikut membutuhkan terjemahan dari bahasa {SOURCE} ke bahasa {TARGET}:\n{INPUT}',
    'Teks: {INPUT}\nTolong beri terjemahan ke bahasa {TARGET} dengan tepat.',
    'Mohon bantu menerjemahkan teks berikut dari bahasa {SOURCE} ke bahasa {TARGET}.\nTeks: {INPUT}',
    
    # Variasi dengan Konteks Pengguna
    'Jika Anda membaca teks berikut dalam bahasa {SOURCE}: {INPUT}\nBagaimana Anda akan menyampaikannya dalam bahasa {TARGET}?',
    'Dalam situasi berikut, teks disampaikan dalam bahasa {SOURCE}: {INPUT}\nBagaimana Anda menerjemahkannya ke bahasa {TARGET}?',
    'Misalkan teks berikut dalam bahasa {SOURCE}: {INPUT}\nApa terjemahannya dalam bahasa {TARGET}?',
    'Bayangkan teks berikut adalah bagian dari percakapan dalam bahasa {SOURCE}: {INPUT}\nTolong beri terjemahan ke bahasa {TARGET}.',
    'Jika teks ini digunakan dalam bahasa {SOURCE}: {INPUT}, bagaimana artinya dalam bahasa {TARGET}?',
    
    # Variasi Percakapan
    'Dalam sebuah percakapan, teks berikut disampaikan dalam bahasa {SOURCE}: {INPUT}\nBagaimana Anda akan menerjemahkannya ke bahasa {TARGET}?',
    'Teks berikut adalah bagian dari dialog dalam bahasa {SOURCE}: {INPUT}\nSilakan terjemahkan ke bahasa {TARGET}.',
    'Jika Anda mendengar teks berikut dalam bahasa {SOURCE}: {INPUT}, apa artinya dalam bahasa {TARGET}?',
    'Berikut adalah bagian dari percakapan dalam bahasa {SOURCE}: {INPUT}\nApa artinya dalam bahasa {TARGET}?',
    'Ketika berbicara, teks berikut digunakan dalam bahasa {SOURCE}: {INPUT}\nApa terjemahannya dalam bahasa {TARGET}?',
     
    # Variasi Gaya Narasi
    'Bayangkan teks ini adalah bagian dari novel dalam bahasa {SOURCE}: {INPUT}\nTolong terjemahkan ke bahasa {TARGET}.',
    'Jika teks berikut adalah deskripsi dalam bahasa {SOURCE}: {INPUT}, bagaimana artinya dalam bahasa {TARGET}?',
    'Narasi berikut diberikan dalam bahasa {SOURCE}: {INPUT}\nBagaimana terjemahannya dalam bahasa {TARGET}?',
    
    # Variasi Teks Informal
    'Pesan berikut ditulis dalam bahasa {SOURCE}: {INPUT}\nBagaimana artinya dalam bahasa {TARGET}?',
    'Jika seseorang menulis teks berikut dalam bahasa {SOURCE}: {INPUT}, bagaimana Anda akan menerjemahkannya ke bahasa {TARGET}?',
    'Pesan singkat ini dalam bahasa {SOURCE}: {INPUT}\nTolong terjemahkan ke bahasa {TARGET}.',
    'Dalam percakapan sehari-hari, teks berikut digunakan dalam bahasa {SOURCE}: {INPUT}\nApa artinya dalam bahasa {TARGET}?',
    
    # Variasi dengan Fokus Konteks
    'Diberikan konteks berikut dalam bahasa {SOURCE}: {INPUT}\nApa artinya dalam bahasa {TARGET}?',
    'Dalam situasi berikut, teks ini digunakan dalam bahasa {SOURCE}: {INPUT}\nSilakan terjemahkan ke bahasa {TARGET}.',
    'Teks ini memiliki makna penting dalam bahasa {SOURCE}: {INPUT}\nTolong terjemahkan ke bahasa {TARGET}.',
    'Konsep berikut dijelaskan dalam bahasa {SOURCE}: {INPUT}\nApa terjemahannya ke bahasa {TARGET}?',
    'Teks berikut disampaikan dalam bahasa {SOURCE}: {INPUT}\nMohon terjemahkan ke bahasa {TARGET}.'
]


# prompt_instruction = ['Terjemahkan teks berikut dari bahasa {SOURCE} ke bahasa {TARGET}.\nTeks: {INPUT}\nTerjemahan:',
#                     '{INPUT}\nTerjemahkan teks di atas dari bahasa {SOURCE} ke bahasa {TARGET}.',
#                     'Teks dalam bahasa {SOURCE}: {INPUT}\nApa terjemahannya dalam bahasa {TARGET}?',
#                     'Terjemahkan teks bahasa {SOURCE} berikut ke bahasa {TARGET}.\nTeks: {INPUT}\nTerjemahan:',
#                     'Teks dalam bahasa {SOURCE}: {INPUT}\nTeks dalam bahasa {TARGET}:']

contextual_prompts = [
    "Diberikan sebuah kalimat dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Berikut adalah kalimat dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Anda memiliki kalimat berikut dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Kalimat berikut dalam bahasa {TARGET} diberikan kepada Anda:\n\n{CONTEXT}\n{INSTRUCTION}",
    "Perhatikan kalimat berikut dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}"
]

semantic_prompts = [
    "Diberikan pasangan sinonim dalam bahasa {SOURCE} dan {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Berikut adalah sinonim dalam bahasa {SOURCE} dan {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Sinonim dalam bahasa {SOURCE} dan {TARGET} diberikan sebagai berikut:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Anda memiliki pasangan kata berikut dalam bahasa {SOURCE} dan {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Diberikan kata-kata yang memiliki arti serupa dalam bahasa {SOURCE} dan {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}"
]

keyword_prompts = [
    "Diberikan sebuah kalimat dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Kalimat berikut diberikan dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Berdasarkan kalimat berikut dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Berikut adalah sebuah kalimat dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Perhatikan kalimat berikut dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}"
]

list_group_label_prompts = [
    "Diberikan daftar kategori kata dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Berikut adalah kategori kata dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Kategori kata berikut diberikan dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Diberikan sebuah daftar kategori dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}",
    "Anda memiliki kategori kata berikut dalam bahasa {TARGET}:\n{CONTEXT}\n\n{INSTRUCTION}"
]
