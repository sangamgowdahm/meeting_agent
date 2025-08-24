import requests
import os
import json
import re
from datetime import datetime,timedelta

class TodoListTools:
  def __init__(self,api_token):
    self.api_token = api_token
    self.base_url = "https://api.todoist.com/rest/v2/"
    self.headers = {
        "Authorization": f"Bearer {self.api_token}",
        "Content-Type": "application/json "
    }

  def get_projects(self):
    """
    Fetches all projects from Todoist.
    """
    response = requests.get(f"{self.base_url}projects", headers=self.headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch projects: {response.status_code} - {response.text}")  
  
  def get_projects(self,project_name):
     
     projects= self.get_projects()

     if "error" in projects:
        return projects
     
     for project in projects:
         if project["name"].lower() == project_name.lower():
             return project
     return None    
  
  def create_project(self, project_name, color="berry-red"):
    """Creates new project if not there"""
    existing_project = self.get_projects(project_name)
    if existing_project:
        return existing_project  # Return existing project if found

    data = {
        "name": project_name,
        "color": color
    }

    response = requests.post(
        f"{self.base_url}/projects",
        headers=self.headers,
        data=json.dumps(data)
    )

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"Failed to create project: {response.status_code} - {response.text}"
        }
  

  def get_collaborators(self,project_id):
    
    """
    Fetches collaborators for a given project.
    """
    response = requests.get(f"{self.base_url}projects/{project_id}/collaborators", headers=self.headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error":f"Failed to fetch collaborators: {response.status_code} - {response.text}"}
  
  def create_task(self,content,project_id,due_string=None,priority=3,assignee_id=None):
    
    """
    Creates a new task in the specified project.
    """
    data = {
        "content": content,
        "project_id": project_id,
        "priority": priority
    }

    if due_string:
        data["due"] = {"string": due_string}

    if assignee_id:
        data["assignee"] = assignee_id

    response = requests.post(f"{self.base_url}tasks", headers=self.headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to create task: {response.status_code} - {response.text}"}
     
  def create_and_assign_task(self,content,project_name,assignee_name=None,due_string=None,priority=3):
    """
    Creates a new task in the specified project and assigns it to a collaborator.
    """
    project = self.get_projects(project_name)
    if not project:
        project= self.create_project(project_name)
        return {"error": "Project not found."}
    
    project_id = project["id"]

    assignee_id = None

    if assignee_name:
       collaborators = self.get_collaborators(project_id)
       if "error" not in collaborators:
          for collaborator in collaborators:
              if collaborator["name"].lower() == assignee_name.lower():
                  assignee_id = collaborator["id"]
                  break
              

    return self.create_task(content, project_id, due_string, priority, assignee_id)        



class TranscriptExtractor:
   def __init__(self,source_type='google_meet'):
      self.source_type = source_type

   def get_transcript(self,meeting_id):
      if self.source_type == 'google_meet':
         return self.get_google_meet_transcript(meeting_id)
      elif self.source_type == 'whatsapp':
         return self.get_whatsapp_transcript(meeting_id)
      elif self.source_type == 'telegram':
         return self.get_telegram_transcript(meeting_id)
    

      else:
         return {"error": "Unsupported source type."}   

   def get_google_meet_transcript(self,meeting_id):
    
      """ get transcript from google meet"""

      return{
      "meeting_id":meeting_id,
      "transcript":"This is a sample transcript from Google Meet meeting.we need to create a new project called marketing campaign and assign tasks to the team, ","participants":["Alice","Bob","Charlie"]
      
      }
   
   def get_whatsapp_transcript(self,chat_id):
      """ get transcript from whatsapp"""

      return{
      "chat_id":chat_id,
      "transcript":"This is a sample transcript from whatsapp meeting.we need to create a new project called marketing campaign and assign tasks to the team, ","participants":["Alice","Bob","Charlie"]
      
      }
   
   def get_telegram_transcript(self,chat_id):
      """ get transcript from whatsapp"""

      return{
      "chat_id":chat_id,
      "transcript":"This is a sample transcript from get_telegram_transcript meeting . we need to create a new project called marketing campaign and assign tasks to the team, ","participants":["Alice","Bob","Charlie"]
      
      }
   


class TelegramCommunicator:
   
   def __init__(self,bot_token,chat_id):
      self.bot_token = bot_token
      self.chat_id = chat_id
      self.base_url = f"https://api.telegram.org/bot{self.bot_token}/"

   def send_message(self,message):
      """ send message to telegram"""

      data={
         "chat_id":self.chat_id,
         "text":message,
         "parse_mode":"Markdown"
         }

      response = requests.post(f"{self.base_url}sendMessage",data=data) 


      if response.status_code == 200:
         return response.json()
      else:
         return {"error": f"Failed to send message: {response.status_code} - {response.text}"}   
      

   def ask_confirmation(self,question):
      """ ask for confirmation with inline keyboard buttons"""

      if options is None:
         options = ["Yes", "No"]

      keyboard=[]

      for option in options:
          keyboard.append([{"text": option, "callback_data": option}])

      data={
          "chat_id":self.chat_id,
          "text":question,
          "reply_markup":json.dumps({"inline_keyboard": keyboard}),
        
          }
      response = requests.post(f"{self.base_url}/sendMessage",data=data)

      if response.status_code == 200:
         return response.json()
      else:
         return {"error": f"Failed to send message: {response.status_code} - {response.text}"}
                

class TaskExtractor:
   def __init__(self,llm):
      self.llm=llm

   def extract_tasks_from_transcript(self, transcript):
    """Extract tasks from transcript using LLM."""

    prompt = f"""Please analyze the following meeting transcript and identify:
    1. Project names mentioned
    2. Tasks that need to be completed
    3. Who should be assigned to each task (if mentioned)
    4. Due dates for each task (if mentioned)

    Format your response as JSON with the following structure:
    {{
      "projects": [
        {{
          "project_name": "Project Name",
          "tasks": [
            {{
              "content": "Task description",
              "assignee": "Assignee name or null",
              "due_string": "Due date string or null",
              "priority": 1-4 (4 is highest priority)
            }}
          ]
        }}
      ]
    }}

    Transcript: {transcript}
    """

    response = self.llm.invoke(prompt).content if self.llm else None
    if not response:
        return {"error": "LLM did not return a response"}

    # Try to extract JSON block (if wrapped in ```json ... ```)
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response  # fallback if raw JSON is returned

    try:
        clean_json = json_str.strip()
        return json.loads(clean_json)
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse JSON: {e}",
            "raw_response": response
        }



class TodoistMeetingManager:
   
   def __init__(self,todoist_api_token,telegram_bot_token=None,telegram_chat_id=None,transcript_source="google_meet",llm=None):
      
      self.todoist = TodoListTools(todoist_api_token)
      self.transcript_extractor = TranscriptExtractor(transcript_source)
      self.telegram = None


      if telegram_bot_token and telegram_chat_id:
         self.telegram = TelegramCommunicator(telegram_bot_token,telegram_chat_id)
         self.task_extractor = TaskExtractor(llm)

   def process_meeting(self, meeting_id):
    """Process meeting and create tasks in Todoist"""

    transcript_data = self.transcript_extractor.get_transcript(meeting_id)
    if "error" in transcript_data:
        return transcript_data

    # Extracted tasks
    extracted_data = self.task_extractor.extract_tasks_from_transcript(transcript_data["transcript"])
    if "error" in extracted_data:
        return extracted_data

    results = {
        "projects_created": [],
        "tasks_created": []
    }

    for project_data in extracted_data['projects']:
        project_name = project_data['project_name']

        # Ask for confirmation via Telegram (optional)
        if self.telegram:
            confirmation = self.telegram.ask_confirmation(
                f"Should I create a new project '{project_name}'?"
            )
            # You could skip project creation here if declined

        project = self.todoist.create_project(project_name)
        if "error" in project:
            results["error"] = project["error"]
            return results

        results["projects_created"].append(project_name)

        # Create tasks under the project
        for task_data in project_data['tasks']:
            task = self.todoist.create_and_assign_task(
                task_data['content'],
                project_name,
                task_data.get('assignee'),
                task_data.get('due_string'),
                task_data.get('priority', 3)
            )

            if 'error' in task:
                results['task_errors'] = results.get('task_errors', []) + [task['error']]
            else:
                results['tasks_created'].append({
                    "content": task_data['content'],
                    "project": project_name,
                    "assignee": task_data.get('assignee')
                })

            # Notify assignee via Telegram
            if self.telegram and task_data.get('assignee'):
                self.telegram.send_message(
                    f"*New Task Assigned*\n\n"
                    f"*Project:* {project_name}\n"
                    f"*Task:* {task_data['content']}\n"
                    f"*Assignee:* {task_data.get('assignee')}\n"
                    f"*Due:* {task_data.get('due_string', 'Not specified')}\n"
                )

    return results
      
     