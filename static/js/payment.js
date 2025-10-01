<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Page</title>
    <link rel="stylesheet" href="/static/style.css"> <!-- Optional: Link to a CSS file for styling -->
</head>
<body>
    <h1>Payment Page</h1>

    <div id="user-info">
        <p>Welcome, <span id="username"></span></p>
        <p>Email: <span id="email"></span></p>
    </div>

    <h2>Payment Information</h2>
    <form id="payment-form">
        <label for="card">Card Number:</label>
        <input type="text" id="card" name="card" required><br><br>

        <label for="expiry">Expiry Date (MM/YY):</label>
        <input type="text" id="expiry" name="expiry" required><br><br>

        <button type="submit">Submit Payment</button>
    </form>

    <script src="/static/js/payment.js"></script> <!-- Link to external JavaScript -->
</body>
</html>
