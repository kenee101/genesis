import os
import streamlit as st
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

def google_login():
        # Define the scope for Google OAuth (accessing user's profile)
        SCOPES = ['https://www.googleapis.com/auth/userinfo.profile']

        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first time.
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        return creds
    
def login_page():
    # Authenticate the user
    creds = google_login()

    # Show the authenticated user profile information
    if creds:
        st.write(f"Authenticated successfully!")
        st.write(f"Token Info: {creds}")
        st.session_state.authenticated = True

    # Fetching user info
    def get_user_info(creds):
        service = build('oauth2', 'v2', credentials=creds)
        user_info = service.userinfo().get().execute()
        return user_info

    user_info = get_user_info(creds)
    st.write(f"User Info: {user_info}")