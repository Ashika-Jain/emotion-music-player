import React, { useState, useContext } from "react";
import {
  Button,
  TextField,
  Typography,
  Paper,
  CircularProgress,
  IconButton,
  InputAdornment,
} from "@mui/material";
import { Visibility, VisibilityOff } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { DarkModeContext } from "../../context/DarkModeContext";

const Register = () => {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  // Access the dark mode state from the context
  const { isDarkMode } = useContext(DarkModeContext);

  const handleRegister = async () => {
    setLoading(true);
    setError(""); // Clear any previous errors

    if (!username || !email || !password || !confirmPassword) {
      setError("Please fill in all fields");
      setLoading(false);
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match");
      setLoading(false);
      return;
    }

    try {
      // Sending username, email, and password to the backend
      const response = await axios.post(
        "http://localhost:8000/users/register/",
        {
          username,
          email,
          password,
        },
        {
          headers: {
            'Content-Type': 'application/json',
          }
        }
      );

      console.log("Registration response:", response.data);

      if (response.status === 201 && response.data.tokens) {
        // Store the tokens in localStorage
        localStorage.setItem("user", JSON.stringify({
          username: response.data.user.username,
          email: response.data.user.email,
          access: response.data.tokens.access,
          refresh: response.data.tokens.refresh
        }));

        // Show success message
        setError(""); // Clear any errors
        alert("Registration successful! Welcome to Moodify!");

        // Redirect to home page since we're already authenticated
        navigate("/home");
      } else {
        setError("Registration failed. Please try again.");
      }

      setLoading(false);
    } catch (error) {
      console.error("Registration failed:", error);
      
      if (error.response) {
        console.error("Error response:", error.response.data);
        setError(error.response.data.error || "Registration failed. Please try again.");
      } else if (error.request) {
        console.error("Error request:", error.request);
        setError("Network error. Please check your connection.");
      } else {
        setError("Registration failed. Please try again later.");
      }

      setLoading(false);
    }
  };

  // Handle "Enter" key press to submit the form
  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      handleRegister(); // Call handleRegister when Enter is pressed
    }
  };

  const handleTogglePasswordVisibility = () => {
    setShowPassword((prev) => !prev);
  };

  const handleToggleConfirmPasswordVisibility = () => {
    setShowConfirmPassword((prev) => !prev);
  };

  const styles = getStyles(isDarkMode); // Dynamically get styles based on dark mode

  return (
    <div style={styles.container}>
      <Paper elevation={4} style={styles.formContainer}>
        <Typography
          variant="h4"
          align="center"
          sx={{
            mb: 3,
            fontFamily: "Poppins",
            color: isDarkMode ? "#ffffff" : "#000000",
          }} // Dynamic color
        >
          Register
        </Typography>
        
        {error && (
          <Typography 
            variant="body2" 
            align="center" 
            sx={{ 
              color: "error.main", 
              mb: 2,
              fontFamily: "Poppins"
            }}
          >
            {error}
          </Typography>
        )}
        
        <TextField
          label="Username"
          variant="outlined"
          fullWidth
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          onKeyPress={handleKeyPress} // Add key press handler
          sx={{ mb: 2 }}
          InputProps={{
            style: {
              fontFamily: "Poppins",
              fontSize: "16px",
              color: isDarkMode ? "#ffffff" : "#000000",
            }, // Dynamic text color
          }}
          InputLabelProps={{
            style: {
              fontFamily: "Poppins",
              color: isDarkMode ? "#cccccc" : "#000000",
            }, // Dynamic label color
          }}
        />
        <TextField
          label="Email"
          variant="outlined"
          fullWidth
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          onKeyPress={handleKeyPress} // Add key press handler
          sx={{ mb: 2 }}
          InputProps={{
            style: {
              fontFamily: "Poppins",
              fontSize: "16px",
              color: isDarkMode ? "#ffffff" : "#000000",
            }, // Dynamic text color
          }}
          InputLabelProps={{
            style: {
              fontFamily: "Poppins",
              color: isDarkMode ? "#cccccc" : "#000000",
            }, // Dynamic label color
          }}
        />
        <TextField
          label="Password"
          type={showPassword ? "text" : "password"}
          variant="outlined"
          fullWidth
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyPress={handleKeyPress} // Add key press handler
          sx={{ mb: 2 }}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  aria-label="toggle password visibility"
                  onClick={handleTogglePasswordVisibility}
                  edge="end"
                  sx={{ color: isDarkMode ? "white" : "#333" }}
                >
                  {showPassword ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
            style: {
              fontFamily: "Poppins",
              fontSize: "16px",
              color: isDarkMode ? "#ffffff" : "#000000",
            }, // Dynamic text color
          }}
          InputLabelProps={{
            style: {
              fontFamily: "Poppins",
              color: isDarkMode ? "#cccccc" : "#000000",
            }, // Dynamic label color
          }}
        />
        <TextField
          label="Confirm Password"
          type={showConfirmPassword ? "text" : "password"}
          variant="outlined"
          fullWidth
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          onKeyPress={handleKeyPress} // Add key press handler
          sx={{ mb: 2 }}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  aria-label="toggle confirm password visibility"
                  onClick={handleToggleConfirmPasswordVisibility}
                  edge="end"
                  sx={{ color: isDarkMode ? "white" : "#333" }}
                >
                  {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
            style: {
              fontFamily: "Poppins",
              fontSize: "16px",
              color: isDarkMode ? "#ffffff" : "#000000",
            }, // Dynamic text color
          }}
          InputLabelProps={{
            style: {
              fontFamily: "Poppins",
              color: isDarkMode ? "#cccccc" : "#000000",
            }, // Dynamic label color
          }}
        />
        <Button
          variant="contained"
          color="primary"
          fullWidth
          onClick={handleRegister}
          sx={{ mb: 2, backgroundColor: "#ff4d4d", font: "inherit" }}
          disabled={loading}
        >
          {loading ? (
            <CircularProgress size={24} color="inherit" />
          ) : (
            "Register"
          )}
        </Button>
        <Typography
          variant="body2"
          align="center"
          sx={{
            cursor: "pointer",
            textDecoration: "underline",
            fontFamily: "Poppins",
            color: isDarkMode ? "#ffffff" : "#000000", // Dynamic color
            "&:hover": {
              color: "#ff4d4d",
              transition: "color 0.2s",
            },
          }}
          onClick={() => navigate("/login")}
        >
          Already have an account? Login
        </Typography>
      </Paper>
    </div>
  );
};

// Function to dynamically return styles based on dark mode
const getStyles = (isDarkMode) => ({
  container: {
    height: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: isDarkMode ? "#121212" : "#f9f9f9",
    transition: "background-color 0.3s ease, color 0.3s ease",
  },
  formContainer: {
    padding: "30px",
    width: "350px",
    borderRadius: "10px",
    boxShadow: "0px 4px 15px rgba(0, 0, 0, 0.2)",
    backgroundColor: isDarkMode ? "#1f1f1f" : "white", // Dynamic form background color
    color: isDarkMode ? "#ffffff" : "#000000", // Dynamic text color
    transition: "background-color 0.3s ease, color 0.3s ease",
  },
});

export default Register;
