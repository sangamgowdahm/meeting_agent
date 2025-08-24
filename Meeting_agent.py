import os
import tempfile 

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from crewai import Agent, Task, Crew
from crewai.process import Process

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader 

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from utils import (
    TodoListTools,
    TranscriptExtractor,
    TelegramCommunicator,
    TaskExtractor,
    TodoistMeetingManager,
)

from langchain_deepseek import ChatDeepSeek
 
from langchain_openai import ChatOpenAI       # for OpenRouter override
from langchain.chains import RetrievalQA


print("All imports successful")


def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "setup": None,
        "deepseek_api_key": None,
        "prepared": False,
        "vectorstore": None,
        "context_analysis": None,
        "meeting_strategy": None,
        "executive_brief": None,
        "todoist_api_key": None,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "transcript_source": "google_meet",
        "meeting_id": "",
        "todoist_manager": None,
        "task_extraction_results": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def process_documents(base_context, uploaded_files):
    """Process base context and uploaded documents"""

    docs = []

    # Save base context as temporary text file
    with tempfile.NamedTemporaryFile(delete=True, mode="w+t", suffix=".txt") as temp:
        temp.write(base_context)
        temp.flush()
        docs.extend(TextLoader(temp.name).load())

    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            suffix = file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=True, suffix=f".{suffix}") as tmp:
                tmp.write(file.getbuffer())
                tmp.flush()

                try:
                    loader = (
                        PyPDFLoader(tmp.name) if suffix.lower() == "pdf" else TextLoader(tmp.name)
                    )
                    docs.extend(loader.load())
                    st.success(f"Loaded {file.name} successfully")
                except Exception as e:
                    st.error(f"Error loading {file.name}: {e}")

    return docs 

def create_vectorstore(docs):
    """Create a FAISS vectorstore from documents"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(api_key=st.session_state["deepseek_api_key"])
    return FAISS.from_documents(splits, embeddings)


def run_crewai_analysis(setup, llm):
    """Run CrewAI analysis for meeting preparation."""

    attendees_text = "\n".join([f"- {attendee}" for attendee in setup["attendees"]])

    # Define agents
    context_agent = Agent(
        name="Context Analyst",
        llm=llm,
        goal="Provide comprehensive analysis of the meeting context",
        backstory=(
            "You are an expert meeting analyst who specializes in preparing context documents for meetings. "
            "You thoroughly research companies and identify key stakeholders."
        ),
        verbose=True,
    )

    strategy_agent = Agent(
        name="Meeting Strategist",
        llm=llm,
        goal="Create a detailed meeting strategy and agenda",
        backstory=(
            "You are a seasoned meeting facilitator who excels at structuring effective business discussions. "
            "You understand how to allocate time optimally."
        ),
        verbose=True,
    )

    brief_agent = Agent(
        name="Executive Briefer",
        llm=llm,
        goal="Generate an executive briefing with actionable insights and recommendations",
        backstory=(
            "You are a master communicator who specializes in crafting executive briefings. "
            "You distill complex information into clear and concise summaries that drive action."
        ),
        verbose=True,
    )

    # Define tasks
    context_task = Task(
        description=f"""
        Analyze the context for the meeting with **{setup['company']}**.
        Consider:
        1. Company background and market position  
        2. Meeting objectives: {setup['objectives']}  
        3. Attendees:  
        {attendees_text}  
        4. Focus areas: {setup['focus']}  

        **FORMAT:** Markdown with clear headings
        """,
        agent=context_agent,
        expected_output="""
        A markdown-formatted context analysis with sections for:
        - Executive Summary  
        - Company Background  
        - Situation Analysis  
        - Key Stakeholders  
        - Strategic Considerations  
        """,
    )

    strategy_task = Task(
        description=f"""
        Develop a detailed meeting strategy and agenda for the {setup['duration']}-minute meeting with **{setup['company']}**.  
        Include:
        1. Time-boxed agenda with specific allocations  
        2. Key talking points for each section  
        3. Discussion questions and role assignments  

        **FORMAT:** Markdown with clear headings
        """,
        agent=strategy_agent,
        expected_output="""
        A markdown-formatted meeting strategy with sections for:
        - Meeting Overview  
        - Detailed Agenda  
        - Key Talking Points  
        - Success Criteria  
        """,
    )

    brief_task = Task(
        description=f"""
        Create an executive briefing for the meeting with **{setup['company']}**.  
        Include:
        1. Executive summary with key points  
        2. Key talking points and recommendations  
        3. Anticipated questions and prepared answers  

        **FORMAT:** Markdown with clear headings
        """,
        agent=brief_agent,
        expected_output="""
        A markdown-formatted executive briefing with sections for:
        - Executive Summary  
        - Key Talking Points  
        - Recommendations  
        - Anticipated Questions and Prepared Answers  
        """,
    )

    # Crew execution
    crew = Crew(
        agents=[context_agent, strategy_agent, brief_agent],
        tasks=[context_task, strategy_task, brief_task],
        verbose=True,
        process=Process.sequential
    )

    return crew.kickoff()


def extract_content(result_item):
    """Extract content from CrewAI result item"""
    if hasattr(result_item,'result'):
        return result_item.result
    if isinstance(result_item,dict)and 'result' in result_item:
        return result_item['result']
    if isinstance(result_item,str):
        return result_item
    return str(result_item)




def fallback_analysis(setup,llm):
    """"Fallback method if crewai fails"""
    attendees_text = "\n".join([f"- {attendee}" for attendee in setup["attendees"]])

    # Context Analysis
    context_prompt=f""" Analyze the context for the meeting with {setup['company']}:
    -Meeting objective :{setup['objective']}.
    -Attendees:
    {attendees_text}
    -Focus areas:{setup['focus']}

    Provide a markdown formatted context analysis with sections for:"""

    strategy_prompt=f"""create a meeting strategy for the {setup['duration']}
    -minute meeting with {setup['company']}:
    -Meeting objective :{setup['objective']}.
    _focus areas:{setup['focus']}

    Provide a markdown formatted meeting strategy with sections for:"""

    brief_prompt=f"""create an executive briefing for the meeting with {setup['company']}:
    -Meeting objective :{setup['objective']}.
    -Focus areas:{setup['focus']}

    Provide a markdown formatted executive briefing with sections for:"""
    
    context_content=llm.invoke(context_prompt).content
    strategy_content=llm.invoke(strategy_prompt).content
    brief_content=llm.invoke(brief_prompt).content

    return context_content,strategy_content,brief_content


# Q-A assit

def create_qa_chain(vectorstore, api_key):
    """create a QA chain for answering questions"""

    prompt_template=PromptTemplate(
        input_variables=["context","question"],
        template="""
        Use the following context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Context: {context}
        Question: {question}
        Answer:
        """,
    )

    retriever=vectorstore.as_retriever(search_kwargs={"k":3})

    llm = ChatDeepSeek(
        model="deepseek-chat",      # DeepSeek V3 model
        api_key=api_key,
        temperature=0
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    

def send_telegram_notification(telegram_bot_token,telegram_chat_id,result):
    """send notification via telegram"""

    telegram = TelegramCommunicator(telegram_bot_token,telegram_chat_id)

    summary = f"*Meeting task summary*\n\n"
    summary+= f"Projects :{', '.join(result['projects_created'])}\n\n"
    summary+= f"Tasks created :{len(result['tasks_created'])}\n\n"

    for task in result['tasks_created']:
        summary+= f"- {task['content']} (Project:{task['project']})"
        if task.get("assignee"):
            summary+= f", Assigned to: {task['assignee']}"
        summary+=")\n"

    return telegram.send_message(summary)


def process_transcript(transcript, todolist_manager):
    """Process a transcript and extract tasks"""

    task_extractor = TaskExtractor(todolist_manager.task_extractor.llm)
    extracted_data = task_extractor.extract_tasks_from_transcript(transcript)

    if "error" in extracted_data:
        return {"error": extracted_data["error"]}
    
    results = {
        "projects_created": [],
        "tasks_created": []
    }

    for project_data in extracted_data.get("projects", []):
        project_name = project_data["name"]
        project = todolist_manager.todolist_tools.create_project(project_name)

        if "error" in project:
            return {"error": project["error"]}

        results["projects_created"].append(project_name)

        for task_data in project_data.get("tasks", []):
            task = todolist_manager.todolist_tools.create_and_assign_task(
                task_data["content"],
                project_name,
                task_data.get("assignee"),
                task_data.get("due_string"),
                task_data.get("priority", 3)
            )

            if "error" in task:
                results["task_errors"] = results.get("task_errors", []) + [task["error"]]
            else:
                results["tasks_created"].append({
                    "content": task_data["content"],
                    "project": project_name,
                    "assignee": task_data.get("assignee"),
                })

    return results




def main():
    st.set_page_config(page_title="Meeting Assistant", layout="wide",page_icon="🤖")
    st.title("🤖 Meeting Assistant with DeepSeek")
    initialize_session_state

    ## we are creating sidebar for api keys and telegram credentials

    with st.sidebar:
        
        openai_api_key=st.text_input(
            "Enter your OpenRouter API Key (DeepSeek)",
            type="password",
            value=st.session_state["deepseek_api_key"],
        )
        if openai_api_key:
            st.session_state["deepseek_api_key"]=openai_api_key
            os.environ["OPENROUTER_API_KEY"]=openai_api_key

        todoist_api_key=st.text_input(
            "Enter your Todoist API Key",
            type="password",
            value=st.session_state["todoist_api_key"],
        )
        if todoist_api_key!=st.session_state["todoist_api_key"]:
            st.session_state["todoist_api_key"]=todoist_api_key
            st.session_state["todoist_manager"]=None


        with st.expander("Telegram integration(Optional)"):
            telegram_bot_token=st.text_input("telegram Bot Token",type="password",value=st.session_state["telegram_bot_token"])
            telegram_chat_id=st.text_input("Telegram Chat ID",value=st.session_state["telegram_chat_id"])

            if telegram_bot_token!=st.session_state["telegram_bot_token"]:
                st.session_state["telegram_bot_token"]=telegram_bot_token
                telegram_credentials_changed=True

            if telegram_chat_id!=st.session_state["telegram_chat_id"]:
                st.session_state["telegram_chat_id"]=telegram_chat_id
                telegram_credentials_changed=True

        st.info("this app helps you prepare for meetings, analyze context, create strategies, and extract tasks from meeting transcripts using DeepSeek and CrewAI")                     
  

    ### we have four tabs in the main area 

    tab_setup,tab_results,tab_qa,tab_tasks=st.tabs(["Meeting setup","Preparation Results","Q&A Assistant","Task Management"])


 ## this is the first tab for meeting setup with all the inputs
    with tab_setup:
        st.subheader("Meeting configuration")
        company_name=st.text_input("company name")
        meeting_objective=st.text_area("Meeting Objective")
        meeting_date=st.date_input("Meeting Date")
        meeting_duration=st.slider("Meeting duration (minutes)",15,180,30)

        st.subheader("Attendees")
        attendees_data=st.data_editor(
            pd.DataFrame({"Name":[""],"Role":[""],"company":[""]}),
            num_rows="dynamic",
            use_container_width=True,
        )

        focus_areas=st.text_area("focus areas or concerns")

        uploaded_files=st.file_uploader(
            "Upload relevant documents (PDF or TXT)",type=["pdf","txt"],accept_multiple_files=True,
            type=["pdf","txt"],accept_multiple_files=True
        )
        if st.button("Prepare Meeting",type="primary",use_container_width=True):
            if not openai_api_key or not company_name or not meeting_objective:
                st.error("Please provide OpenRouter API key, company name, and meeting objective")
            else:
                attendees_formatted=[]
                for _,row in attendees_data.iterrows():
                    if row["Name"].strip():
                        
                        attendees_formatted.append(f"{row['Name'].strip()} ,{row['Role'].strip()} from {row['company'].strip()}")

                st.session_state["setup"]={
                    "company":company_name,
                    "objectives":meeting_objective,
                    "date":str(meeting_date),
                    "duration":meeting_duration,
                    "attendees":attendees_formatted,
                    "focus":focus_areas,
                    "files":uploaded_files,
                }
                st.session_state["prepared"]=False
                st.rerun()        

    with tab_results:
        if st.session_state["setup"] and not st.session_state["prepared"]:
            with st.status("processing meeting data",expanded=True)as status:
                status:
                    setup=st.session_state["setup"]
                
                    attendees_text="\n".join([f"- {attendee}" for attendee in setup["attendees"]])
                base_context=f"""
                Meeting Information:
                -Company: {setup['company']}
-Objectives: {setup['objectives']}
-Date: {setup['date']}
-Duration: {setup['duration']} minutes
-Focus Areas: {setup['focus']}

Attendees:
{attendees_text}
"""
                docs=process_documents(base_context,setup["files"])

                vectorstore = create_vectorstore(docs)
                st.session_state["vectorstore"]=vectorstore

                llm=ChatDeepSeek(
                    model="deepseek-chat",
                    api_key=st.session_state["deepseek_api_key"],
                    temperature=0.3,
                )
                try:
                    results=run_crewai_analysis(setup,llm)

                    if isinstance(results,list)and len(results)>=3:
                        context_analysis=extract_content(results[0])
                        strategy_content=extract_content(results[1])
                        brief_content=extract_content(results[2])
                    else:
                        raise Exception ("")    




            
                            