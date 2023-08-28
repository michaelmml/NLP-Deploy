# Importing necessary libraries
import streamlit as st
import pdfplumber
import numpy as np
import pandas as pd
from dateutil.parser import parse

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
                    df_speaker.to_csv(os.path.join(path, ticker + '_' + date + '_speakers.csv'), index=False)

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

    print(pdf.pages[0].extract_text())
    data = data.append(df)
    return data

# Streamlit UI
st.title("PDF Text Extractor")
ticker = st.text_input("Type ticker corresponding to the document:", value='ALK')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.write("Extracting text... Please wait.")
    table = extract_text_from_pdf(uploaded_file, ticker)
    
    if extracted_text:
        st.subheader("Extracted Text:")
        st.table(table)
    else:
        st.write("The PDF doesn't contain any recognizable text.")


