# Importing necessary libraries
import streamlit as st
import pdfplumber
import numpy as np
import pandas as pd
from dateutil.parser import parse
import re

def extract_text_from_pdf(file, ticker):
    # Extract text from all pages
    # full_text = ""
    # for page in pdf.pages:
    # full_text += page.extract_text()

    found_start = False
    out = ''
    columns = ['particip', 'qna', 'transcript', 'date', 'company']
    data = pd.DataFrame(columns=columns)
  
    with pdfplumber.open(file) as pdf:

            for i,v in enumerate(pdf.pages):
                curr_page = pdf.pages[i]
                txt = curr_page.extract_text()

                if i == 0:
                    if txt.find(ticker + ' -') != -1:
                        dt_idx = txt.find(ticker + ' -')
                        qt_txt = txt[dt_idx:].split('\n')
                        qt_txt = [x for x in qt_txt if x != ' '][0].strip(ticker + ' -')  # remove ' '
                    else:
                        qt_txt = [x for x in txt.split('\n') if 'Earnings' in x][0]
                    quarter = qt_txt.split()[0]
                    year = qt_txt.split()[1]
                    date_idx = txt.find('EVENT DATE/TIME: ') + len('EVENT DATE/TIME: ')
                    date_idx_end = date_idx + 50  # static character count for length of date
                    date = parse(txt[date_idx:date_idx_end].split('\n')[0]).strftime('%m-%d-%y')
                    comp_name = qt_txt.split(' Earnings Call')[0][qt_txt.split(' Earnings Call')[0].find(year)+ len(year) + 1:]

                elif i == 1:
                    corp_idx = txt.find('CORPORATE PARTICIPANTS')+len('CORPORATE PARTICIPANTS\n')
                    corp_idx_end = txt.find('\nCONFERENCE CALL PARTICIPANTS')
                    conf_idx = txt.find('CONFERENCE CALL PARTICIPANTS') + len('CONFERENCE CALL PARTICIPANTS\n')
                    conf_idx_end = txt.find('PRESENTATION')
                    corp_txt = txt[corp_idx:corp_idx_end].split('\n')
                    corp_name = [x.split(comp_name.split()[0])[0] for x in corp_txt]
                    corp_name = [x.strip() for x in corp_name if x.strip() != '']
                    corp_speaker1 = corp_name
                    corp_speaker2 = [x.split()[0] + ' ' + x.split()[1] for x in corp_txt if len(x.split()) >= 2]
                    corp_speaker3 = [x.split()[0] + ' ' + x.split()[2] for x in corp_txt if
                                     len(x.split()) >= 3]
                    corp_speaker4 = [x.split()[0] + ' ' + x.split()[1] + ' ' + x.split()[2] for x in corp_txt if
                                     len(x.split()) >= 3]

                    conf_txt = txt[conf_idx:conf_idx_end].split('\n')
                    conf_speaker1 = [x.split()[0] + ' ' + x.split()[1] for x in conf_txt if len(x.split()) >= 2]
                    conf_speaker2 = [x.split()[0] + ' ' + x.split()[2] for x in conf_txt if len(x.split()) >= 3]    # accounts for speakers with middle name
                    conf_speaker3 = [x.split()[0] + ' ' + x.split()[1] + ' ' + x.split()[2] for x in conf_txt if len(x.split()) >=3]    # accounts for speakers with middle name

                    speaker_list = corp_speaker1 + corp_speaker2 + corp_speaker3 + corp_speaker4 + conf_speaker1 + conf_speaker2 + conf_speaker3

                    columns = ['corp_particip','conf_particip']
                    df_speaker = pd.DataFrame(columns=columns)
                    corp_list = corp_speaker1 + corp_speaker2 + corp_speaker3 + corp_speaker4
                    conf_list = conf_speaker1 + conf_speaker2 + conf_speaker3
                    df_speaker['corp_particip'] = corp_list + (max(len(corp_list), len(conf_list)) -len(corp_list)) * ['']
                    df_speaker['conf_particip'] = conf_list + (max(len(corp_list), len(conf_list)) -len(conf_list)) * ['']
                    # df_speaker.to_csv(os.path.join(path, ticker + '_' + date + '_speakers.csv'), index=False)

                # remove footer
                if i != 0 and i != len(pdf.pages)-1:
                    beginning_idx = txt.find('Earnings Call')+len('Earnings Call\n')
                    pg_num_idx = txt.find('\n' + str(i + 1))
                    txt = txt[beginning_idx:pg_num_idx]

                # remove disclaimer
                elif i ==len(pdf.pages)-1:  # if last page remove disclaimer
                    beginning_idx = txt.find('Earnings Call')+len('Earnings Call\n')
                    dis_idx = txt.find('DISCLAIMER')
                    txt = txt[beginning_idx:dis_idx]

                # write to file
                if found_start == True:
                    out = out+txt

                elif txt.find('PRESENTATION') != -1 and found_start == False:
                    found_start = True
                    start_idx = txt.find('PRESENTATION')+len('PRESENTATION\n')
                    out = out+txt[start_idx:]

    columns = ['particip', 'qna', 'transcript', 'date', 'company']
    df = pd.DataFrame(columns=columns)
    speaker = ''
    content = ''
    qna_start_j = 1000  # initialize
    j = 0
    for line in out.split('\n'):

            if line.find('QUESTIONS AND ANSWERS') != -1:
                qna_start_j = j + 2
            if len(line.split()) == 1:
                find_speaker = [x for x in ['Operator'] if (x in line)]
            else:
                find_speaker = [x for x in speaker_list if (x + ' -' in line)]  # + ' -'

            if find_speaker == []:
                content = content + ' ' + line
            else:
                if speaker != '':
                    if speaker in corp_list:
                        df.loc[j] = [speaker, 0, content, date, ticker]
                    else:
                        df.loc[j] = [speaker, 1, content, date, ticker]
                    content = ''
                    j += 1

                speaker = find_speaker[0]

    return df

# Function to extract sentences with keywords
def extract_sentences_with_all_sequences(df, sequences):
    # Filter rows where 'qna' is 0
    org_df = df[df['qna'] == 0]
    
    # Store results
    results = []
    
    for _, row in org_df.iterrows():
        # Split transcript into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', row['transcript'])
        for sentence in sentences:
            # Check if all sequences are in the sentence
            if all(seq in sentence for seq in sequences):
                results.append({
                    'particip': row['particip'],
                    'sentence': sentence.strip()
                })
    
    return pd.DataFrame(results)

############# NLP Functions

def clean(text):
    text = re.sub('[0-9]+.\t', '', str(text)) # removing paragraph numbers
    text = re.sub('\n ', '', str(text))
    text = re.sub('\n', ' ', str(text))
    text = re.sub("'s", '', str(text))
    text = re.sub("-", ' ', str(text))
    text = re.sub("— ", '', str(text))
    text = re.sub('\"', '', str(text))
    text = re.sub("Mr\.", 'Mr', str(text))
    text = re.sub("Mrs\.", 'Mrs', str(text))
    text = re.sub("[\]]", "", str(text))
    return text

def count_stopwords(text, stopwords):
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stopwords]
    return len(stopwords_x)

############# Plotting Functions

def plot_boxplots(data, plot_vars, labels, figsize):
   # We need to identify is this a matrix or a vector
    if plot_vars.ndim == 1:
        nrows = 1
        ncols = plot_vars.shape[0]
    else:
        nrows= plot_vars.shape[0]
        ncols = plot_vars.shape[1]

    f, axes = plt.subplots(nrows, ncols, sharey=False, figsize=(15,5))

    for i in range(nrows):
        for j in range(ncols):
            if plot_vars[i,j]!=None:
                if axes.ndim > 1:

                    axes[i,j].set_title(labels[plot_vars[i,j]])
                    axes[i,j].grid(True)
                    #Set x ticks
                    axes[i,j].tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)
                    # Plot a boxplot for the column in plot_vars
                    axes[i,j].boxplot(data[plot_vars[i,j]])
                else:

                    axes[j].set_title(labels[plot_vars[i,j]])
                    axes[j].grid(True)
                    #Set x ticks
                    axes[j].tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)
                    # Plot a boxplot for the column in plot_vars
                    axes[j].boxplot(data[plot_vars[i,j]])
                
            else:
                axes[i,j].set_visible(False)

    f.tight_layout()
    st.pyplot(plt)

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def plot_histograms(data, plot_vars, xlim, labels, figsize):

    kwargs = dict(hist_kws={'alpha':.7}, kde_kws={'linewidth':2})
    fig, axes = plt.subplots(plot_vars.shape[0], plot_vars.shape[1], figsize=figsize, sharey=False, dpi=100)

    for i in range(plot_vars.shape[1]):

        sns.distplot(data[plot_vars[0,i]] , color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1),), 
                     ax=axes[i], axlabel=labels[plot_vars[0,i]], bins= 50, norm_hist = True)
        # For a better visualization we set the x limit
        axes[i].set_xlim(left=0, right=xlim[i])
        
    fig.tight_layout()
    st.pyplot(plt)

def intro_plot(data_subset):
    labels_dict={'sum_word_count': 'Word Count of Summaries','text_word_count': 'Word Count of Texts',
                 'sum_char_count': 'Char Count of Summaries','text_char_count': 'Char Count of Texts',
                 'sum_word_density': 'Word Density of Summaries','text_word_density': 'Word Density of Texts',
                 'sum_punc_count': 'Punctuation Count of Summaries','text_punc_count': 'Punctuation Count of Texts',
                 'text_sent_count': 'Sentence Count of Texts', 'sum_sent_count': 'Sentence Count of Summaries',
                 'text_sent_density': 'Sentence Density of Texts', 'sum_sent_density': 'Sentence Density of Summaries',
                 'text_stopw_count': 'Stopwords Count of Texts', 'sum_stopw_count': 'Stopwords Count of Summaries',
                 'ADJ': 'adjective','ADP': 'adposition', 'ADV': 'adverb','CONJ': 'conjunction',
                 'DET': 'determiner','NOUN': 'noun', 'text_unknown_count': 'Unknown words in Texts',
                 'sum_unknown_count': 'Unknown words in Summaries',}
    
    data_subset = data[data['qna'] == 0]
    data_subset = data_subset.drop(['particip'], axis=1)
    
    data_subset['text_sent_count'] = data_subset['Transcript_clean'].apply(lambda x : len(split_sentences(x)))
    data_subset['text_word_count'] = data_subset['Transcript_clean'].apply(lambda x : len(x.split()))
    data_subset['text_char_count'] = data_subset['Transcript_clean'].apply(lambda x : len(x.replace(" ","")))
    data_subset['text_word_density'] = data_subset['text_word_count'] / (data_subset['text_char_count'] + 1)
    data_subset['text_sent_density'] = data_subset['text_sent_count'] / (data_subset['text_word_count'] + 1)
    data_subset['text_punc_count'] = data_subset['Transcript_clean'].apply(lambda x : len([a for a in x if a in punc]))
    
    # Stopwords
    data_subset['text_stopw_count'] =  data_subset['Transcript_clean'].apply(lambda x : count_stopwords(x, stopwords))
    data_subset['text_stopw_density'] = data_subset['text_stopw_count'] / (data_subset['text_word_count'] + 1)
    
    plot_vars=np.array([['text_sent_count', 'text_word_count', 'text_char_count','text_sent_density','text_word_density']])
    plot_boxplots(data_subset, plot_vars, labels_dict, figsize=(10,3))
    plot_histograms(data_subset, plot_vars, [200, 3000, 8000, 0.5, 0.5], labels_dict, figsize=(10,3))


# Streamlit UI
st.title("Earnings Transcript Extractor")
st.write("Works best with Refinitiv format, extracts based on structure of the transcript such as the speakers and positioning of the text.")
ticker = st.text_input("Type ticker corresponding to the document:", value='ALK')
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.write("Extracting text... Please wait.")
    table = extract_text_from_pdf(uploaded_file, ticker)
    intro_plot(table)
    # Taking keyword input from user
    # Taking sequences input from user and splitting by comma
    sequences_input = st.text_input("Enter sequences of characters separated by commas")
    sequences = [seq.strip() for seq in sequences_input.split(',')]
    if st.button("Search"):
        if sequences:
            result_df = extract_sentences_with_all_sequences(table, sequences)
            
            if not result_df.empty:
                st.table(result_df)
            else:
                st.write("No sentences found with the provided keywords.")

    st.subheader("Full Transcript:")
    st.write(table)
