import ssl
try:
   ssl.create_default_context()
   print("SSL 模块正常工作。")
except Exception as e:
   print(f"SSL 模块不可用: {e}")