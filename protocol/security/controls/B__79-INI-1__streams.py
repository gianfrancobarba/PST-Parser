def http_hello_page(request, response):
    username = request.read()
    response.write('<html><body>Hello ' + username + '</body></html>')
    response.close()
