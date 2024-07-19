
from supabase import create_client, Client

url: str = "https://bslcdkyoakgwthfynzjb.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJzbGNka3lvYWtnd3RoZnluempiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDY2NTc3MzcsImV4cCI6MjAyMjIzMzczN30.9STCcQclnxoflZ0BXKZm6_0yqtIsvpMTDgOPaApYP9c"
supabase: Client = create_client(url, key)
