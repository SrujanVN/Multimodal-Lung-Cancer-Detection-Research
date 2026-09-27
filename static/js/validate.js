document.addEventListener('DOMContentLoaded', function() {
    const registerForm = document.getElementById('registerForm');
    const loginForm = document.getElementById('loginForm');
    const password = document.getElementById('password');
    const passwordStrength = document.getElementById('passwordStrength');
    const passwordMessage = document.getElementById('passwordMessage');

    if (password) {
        password.addEventListener('input', function() {
            const value = this.value;
            let strength = 0;
            if (value.length >= 8) strength += 1;
            if (/[A-Z]/.test(value)) strength += 1;
            if (/[0-9]/.test(value)) strength += 1;
            if (/[!@#$%^&*]/.test(value)) strength += 1;

            // Update strength class
            passwordStrength.className = 'strength-' + Math.min(strength, 3);

            // Show/hide and update message for weak/medium passwords
            if (strength < 2) {
                passwordMessage.style.display = 'block';
                if (strength === 0) {
                    passwordMessage.textContent = 'Password is too weak. Use at least 8 characters, an uppercase letter, a number, and a special character (!@#$%^&*).';
                } else {
                    passwordMessage.textContent = 'Password is medium. Consider adding more complexity (e.g., special characters or numbers).';
                }
            } else {
                passwordMessage.style.display = 'none';
                passwordMessage.textContent = '';
            }
        });
    }

    if (registerForm) {
        registerForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const firstName = document.getElementById('firstName').value.trim();
            const lastName = document.getElementById('lastName').value.trim();
            const dob = document.getElementById('dob').value;
            const phone = document.getElementById('phone').value.trim();
            const email = document.getElementById('email').value.trim();
            const age = document.getElementById('age').value.trim();
            const sex = document.getElementById('sex').value;
            const password = document.getElementById('password').value.trim();
            const message = document.getElementById('message');

            message.textContent = '';

            if (!firstName || !lastName || !dob || !phone || !email || !age || !sex || !password) {
                message.textContent = 'Please fill all fields.';
                return;
            }

            if (firstName.length > 30 || lastName.length > 30) {
                message.textContent = 'First or last name must not exceed 30 characters.';
                return;
            }

            if (!/^\+?\d{10,15}$/.test(phone)) {
                message.textContent = 'Phone number must be 10-15 digits, optionally starting with +.';
                return;
            }

            if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
                message.textContent = 'Please enter a valid email address.';
                return;
            }

            if (age < 13 || age > 120) {
                message.textContent = 'Age must be between 13 and 120.';
                return;
            }

            if (sex === '') {
                message.textContent = 'Please select a sex.';
                return;
            }

            if (password.length < 8 || !/[A-Z]/.test(password) || !/[0-9]/.test(password) || !/[!@#$%^&*]/.test(password)) {
                message.textContent = 'Password must be at least 8 characters with uppercase, number, and special character (!@#$%^&*).';
                return;
            }

            fetch('/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ firstName, lastName, dob, phone, email, age, sex, password })
            })
            .then(response => {
                if (!response.ok) throw new Error('Server error');
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    window.location.href = '/login';
                } else {
                    message.textContent = data.error || 'Registration failed. Please try again.';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                message.textContent = 'An unexpected error occurred. Please try again.';
            });
        });
    }

    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const username = document.getElementById('username').value.trim();
            const password = document.getElementById('password').value.trim();
            const message = document.getElementById('message');

            message.textContent = '';

            if (!username && !password) {
                message.textContent = 'Please fill both the username and password fields.';
            } else if (!username) {
                message.textContent = 'Please fill the username field.';
            } else if (!password) {
                message.textContent = 'Please fill the password field.';
            } else {
                fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: username, password: password })
                })
                .then(response => {
                    if (!response.ok) throw new Error('Server error');
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        window.location.href = '/dashboard';
                    } else if (data.error === 'account_not_found') {
                        message.textContent = 'Account not found. Please check your username or register.';
                    } else if (data.error === 'username_not_found') {
                        message.textContent = 'Username not found. Please try again or register.';
                    } else if (data.error === 'incorrect_password') {
                        message.textContent = 'Incorrect password. Please try again.';
                    } else {
                        message.textContent = 'Login failed. Please try again later.';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    if (error.message.includes('network')) {
                        message.textContent = 'Network error. Please check your connection.';
                    } else if (error.message.includes('server')) {
                        message.textContent = 'Server is temporarily unavailable. Please try again later.';
                    } else {
                        message.textContent = 'An unexpected error occurred. Please try again.';
                    }
                });
            }
        });
    }
});