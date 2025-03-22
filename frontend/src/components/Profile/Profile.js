import React, { useEffect, useState, useContext } from "react";
import {
  Paper,
  Typography,
  Box,
  CircularProgress,
  Card,
  CardContent,
  Button,
  Avatar,
} from "@mui/material";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { DarkModeContext } from "../../context/DarkModeContext";
import MoodHistory from "./MoodHistory";
import Recommendations from "./Recommendations";

const CACHE_KEY = "userProfileCache";

const timeout = (ms) => {
  return new Promise((_, reject) =>
    setTimeout(() => reject(new Error("Request timed out")), ms),
  );
};

const ProfilePage = () => {
  const [userData, setUserData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [loadingText, setLoadingText] = useState("Loading...");
  const [randomImage, setRandomImage] = useState("");
  const navigate = useNavigate();
  
  // Get user data from localStorage
  const userStr = localStorage.getItem("user");
  const user = userStr ? JSON.parse(userStr) : null;
  const token = user?.access;

  const placeholderImages = [
    require("../../assets/images/profile.webp"),
    require("../../assets/images/OIP.jpg"),
    require("../../assets/images/OIP2.webp"),
    require("../../assets/images/OIP3.png"),
    require("../../assets/images/OIP4.png"),
    require("../../assets/images/OIP5.png"),
    require("../../assets/images/OIP6.webp"),
    require("../../assets/images/OIP7.webp"),
    require("../../assets/images/OIP8.webp"),
    require("../../assets/images/OIP9.webp"),
    require("../../assets/images/OIP10.webp"),
    require("../../assets/images/OIP11.webp"),
    require("../../assets/images/OIP12.webp"),
    require("../../assets/images/OIP13.webp"),
    require("../../assets/images/OIP14.webp"),
    require("../../assets/images/OIP15.webp"),
    require("../../assets/images/OIP16.webp"),
    require("../../assets/images/OIP17.webp"),
    require("../../assets/images/OIP18.webp"),
    require("../../assets/images/OIP19.webp"),
    require("../../assets/images/OIP20.webp"),
  ];

  const { isDarkMode } = useContext(DarkModeContext);

  useEffect(() => {
    // Randomly select an image on component mount
    setRandomImage(
      placeholderImages[Math.floor(Math.random() * placeholderImages.length)],
    );

    if (!user || !token) {
      alert("You are not authenticated. Please log in.");
      navigate("/login");
      return;
    }

    fetchUserData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchUserData = async () => {
    setIsLoading(true);

    try {
      const response = await axios.get(
        "http://localhost:8000/users/user/profile/",
        {
          headers: { Authorization: `Bearer ${token}` },
          timeout: 60000, // 60 seconds timeout
        },
      );

      console.log("Profile Response Data:", response.data);
      console.log("Mood History:", response.data.mood_history);
      
      // Ensure we have arrays for moods and recommendations
      const processedData = {
        ...response.data,
        recent_moods: response.data.recent_moods || [],
        recent_recommendations: response.data.recent_recommendations || []
      };
      
      setUserData(processedData);
      // Update cache with processed data
      localStorage.setItem(CACHE_KEY, JSON.stringify(processedData));
      setError(""); // Clear any existing errors
    } catch (error) {
      console.error("Error fetching user data:", error);
      if (error.response && error.response.status === 401) {
        // Token expired or invalid
        localStorage.removeItem("user");
        navigate("/login");
        return;
      }

      // Use cached data as a fallback
      const cachedUserData = localStorage.getItem(CACHE_KEY);
      if (cachedUserData) {
        setUserData(JSON.parse(cachedUserData));
        console.log(
          "Failed to fetch profile data. Our servers might be down. Please try again later.",
        );
      } else {
        setError(
          "Failed to fetch profile data. Our servers might be down. Please try again later.",
        );
        console.error("No cached profile data available.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleMoodClick = async (mood) => {
    try {
      setLoadingText(`Fetching recommendations for "${mood}"...`);
      setIsLoading(true);
      const response = await Promise.race([
        axios.post(
          "https://moodify-emotion-music-app.onrender.com/api/music_recommendation/",
          { emotion: mood.toLowerCase() }, // Pass the mood as a parameter
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          },
        ),
        timeout(60000),
      ]);

      const { emotion, recommendations } = response.data;
      navigate("/results", { state: { emotion, recommendations } });
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      alert("Failed to fetch recommendations. Please try again later.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteMood = async (index) => {
    // Store the original data at the beginning of the function
    const originalData = { ...userData };

    try {
      const moodToDelete = userData.recent_moods[index];
      console.log("Attempting to delete mood:", moodToDelete);
      
      // Create updated user data with the mood removed
      const updatedMoods = userData.recent_moods.filter((_, i) => i !== index);
      
      // Update UI optimistically
      setUserData({
        ...userData,
        recent_moods: updatedMoods
      });

      // Send the update to the server
      const response = await axios.put(
        "http://localhost:8000/users/user/profile/update/",
        {
          recent_moods: updatedMoods,
          username: userData.username,
          email: userData.email
        },
        {
          headers: { 
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.status === 200) {
        // Update local state with server response
        const updatedData = {
          ...userData,
          recent_moods: response.data.recent_moods || updatedMoods
        };
        setUserData(updatedData);
        localStorage.setItem(CACHE_KEY, JSON.stringify(updatedData));
        console.log("Successfully deleted mood");
      } else {
        throw new Error('Delete operation failed');
      }
    } catch (error) {
      console.error("Error deleting mood:", error);
      if (error.response) {
        console.error("Server response:", error.response.data);
      }
      
      // Revert to the original data on error
      setUserData(originalData);
      localStorage.setItem(CACHE_KEY, JSON.stringify(originalData));
      
      alert("Failed to delete mood. Please try again.");
    }
  };

  const handleDeleteRecommendation = async (index) => {
    // Store the original data at the beginning of the function
    const originalData = { ...userData };

    try {
      const recToDelete = userData.recent_recommendations[index];
      console.log("Attempting to delete recommendation:", recToDelete);
      
      // Create updated user data with the recommendation removed
      const updatedRecommendations = userData.recent_recommendations.filter((_, i) => i !== index);
      
      // Update UI optimistically
      setUserData({
        ...userData,
        recent_recommendations: updatedRecommendations
      });

      // Send the update to the server
      const response = await axios.put(
        "http://localhost:8000/users/user/profile/update/",
        {
          recent_recommendations: updatedRecommendations,
          username: userData.username,
          email: userData.email
        },
        {
          headers: { 
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.status === 200) {
        // Update local state with server response
        const updatedData = {
          ...userData,
          recent_recommendations: response.data.recent_recommendations || updatedRecommendations
        };
        setUserData(updatedData);
        localStorage.setItem(CACHE_KEY, JSON.stringify(updatedData));
        console.log("Successfully deleted recommendation");
      } else {
        throw new Error('Delete operation failed');
      }
    } catch (error) {
      console.error("Error deleting recommendation:", error);
      if (error.response) {
        console.error("Server response:", error.response.data);
      }
      
      // Revert to the original data on error
      setUserData(originalData);
      localStorage.setItem(CACHE_KEY, JSON.stringify(originalData));
      
      alert("Failed to delete recommendation. Please try again.");
    }
  };

  const styles = getStyles(isDarkMode); // Dynamically get styles based on dark mode

  return (
    <Box style={styles.container}>
      {isLoading && (
        <Box sx={styles.loadingOverlay}>
          <CircularProgress sx={{ color: "#ff4d4d" }} />
          <Typography
            variant="h6"
            style={{ marginTop: "10px", color: "white", font: "inherit" }}
          >
            {loadingText}
          </Typography>
          <Typography
            variant="h6"
            style={{
              marginTop: "10px",
              color: "white",
              font: "inherit",
              textAlign: "center",
              fontSize: "14px",
              padding: "0 2rem",
            }}
          >
            Note that our servers might be slow or experience downtime due to
            high traffic, or they may spin down after periods of inactivity. It
            may take up to 2 minutes to process during these times. We
            appreciate your patience, and apologize for any inconvenience.
          </Typography>
        </Box>
      )}

      {isLoading ? null : error ? (
        <Typography variant="h6" color="error" style={{ font: "inherit" }}>
          {error}
        </Typography>
      ) : (
        <Paper elevation={4} style={styles.profileContainer}>
          <Typography variant="h5" style={styles.title}>
            Welcome, {userData.username}!
          </Typography>
          <Box style={styles.infoSection}>
            <Avatar
              alt="User Avatar"
              src={randomImage}
              sx={{
                width: 100,
                height: 100,
                border: "4px solid #ff4d4d",
                margin: "0 auto",
                marginBottom: "20px",
              }}
            />
            <Typography variant="h6" style={styles.text}>
              Your Username: {userData.username}
            </Typography>
            <Typography variant="h6" style={styles.text}>
              Your Email: {userData.email}
            </Typography>
          </Box>

          {/*<Box sx={styles.section}>*/}
          {/*  <Typography variant="h6" style={styles.sectionTitle}>*/}
          {/*    Your Listening History*/}
          {/*  </Typography>*/}
          {/*  {userData.listening_history && userData.listening_history.length > 0 ? (*/}
          {/*    userData.listening_history.map((track, index) => (*/}
          {/*      <Card key={index} style={styles.card}>*/}
          {/*        <CardContent>*/}
          {/*          <Typography variant="body1" style={styles.text}>*/}
          {/*            {track}*/}
          {/*          </Typography>*/}
          {/*        </CardContent>*/}
          {/*      </Card>*/}
          {/*    ))*/}
          {/*  ) : (*/}
          {/*    <Typography variant="body2" style={styles.noData}>*/}
          {/*      No listening history available.*/}
          {/*    </Typography>*/}
          {/*  )}*/}
          {/*</Box>*/}

          <Box sx={styles.section}>
            <Typography variant="h6" style={styles.sectionTitle}>
              Your Recent Moods
            </Typography>
            {console.log("Rendering mood history:", userData?.recent_moods)}
            {Array.isArray(userData?.recent_moods) && userData.recent_moods.length > 0 ? (
              <MoodHistory 
                moods={userData.recent_moods} 
                onDeleteMood={handleDeleteMood}
              />
            ) : (
              <Typography variant="body2" style={styles.noData}>
                No mood history available. Try detecting some emotions!
              </Typography>
            )}
            {userData?.total_moods > 5 && (
              <Typography variant="caption" style={styles.moreInfo}>
                Showing 5 most recent moods out of {userData.total_moods} total
              </Typography>
            )}
          </Box>

          <Box sx={styles.section}>
            <Typography variant="h6" style={styles.sectionTitle}>
              Your Recent Recommendations
            </Typography>
            {userData.recent_recommendations && userData.recent_recommendations.length > 0 ? (
              <Recommendations 
                recommendations={userData.recent_recommendations}
                onDeleteRecommendation={handleDeleteRecommendation}
              />
            ) : (
              <Typography variant="body2" style={styles.noData}>
                No recommendations history available. Try getting some music recommendations!
              </Typography>
            )}
            {userData.total_recommendations > 5 && (
              <Typography variant="caption" style={styles.moreInfo}>
                Showing 5 most recent recommendations out of {userData.total_recommendations} total
              </Typography>
            )}
          </Box>
        </Paper>
      )}
    </Box>
  );
};

// Function to dynamically return styles based on dark mode
const getStyles = (isDarkMode) => ({
  container: {
    height: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    fontFamily: "Poppins, sans-serif",
    padding: "0",
    backgroundColor: isDarkMode ? "#121212" : "#f5f5f5",
    color: isDarkMode ? "#ffffff" : "#000000",
    transition: "background-color 0.3s ease",
  },
  loadingOverlay: {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    zIndex: 1000,
  },
  profileContainer: {
    padding: "30px",
    width: "70%",
    maxHeight: "85vh",
    overflowY: "auto",
    borderRadius: "10px",
    boxShadow: "0px 4px 15px rgba(0, 0, 0, 0.2)",
    backgroundColor: isDarkMode ? "#1f1f1f" : "#ffffff",
    textAlign: "center",
    transition: "background-color 0.3s ease",
  },
  title: {
    marginBottom: "20px",
    fontFamily: "Poppins, sans-serif",
    color: isDarkMode ? "#ffffff" : "#333",
  },
  infoSection: {
    marginBottom: "20px",
    backgroundColor: isDarkMode ? "#333333" : "#ffffff",
    padding: "10px",
    borderRadius: "8px",
    boxShadow: "0px 2px 8px rgba(0, 0, 0, 0.1)",
    transition: "all 0.3s ease",
    color: isDarkMode ? "#ffffff" : "#000000",
  },
  section: {
    marginTop: "15px",
    textAlign: "left",
    padding: "10px",
    color: isDarkMode ? "#ffffff" : "#000000",
  },
  sectionTitle: {
    textDecoration: "underline",
    fontFamily: "Poppins, sans-serif",
    marginBottom: "10px",
    color: isDarkMode ? "#bbbbbb" : "#555",
    fontWeight: 500,
  },
  card: {
    marginBottom: "10px",
    borderRadius: "8px",
    boxShadow: "0px 2px 10px rgba(0, 0, 0, 0.1)",
    padding: "10px",
    backgroundColor: isDarkMode ? "#333333" : "#ffffff",
    transition: "transform 0.3s ease, box-shadow 0.3s ease",
    cursor: "pointer",
    "&:hover": {
      transform: "scale(1.02)",
      boxShadow: "0px 4px 15px rgba(0, 0, 0, 0.2)",
    },
    color: isDarkMode ? "#ffffff" : "#000000",
  },
  text: {
    fontFamily: "Poppins, sans-serif",
    color: isDarkMode ? "#cccccc" : "#000000",
    fontSize: "16px",
  },
  noData: {
    color: isDarkMode ? "#bbbbbb" : "#999",
    fontFamily: "Poppins, sans-serif",
  },
  moodCard: {
    marginBottom: "10px",
    borderRadius: "8px",
    boxShadow: "0px 2px 10px rgba(0, 0, 0, 0.1)",
    backgroundColor: isDarkMode ? "#333333" : "#ffffff",
    cursor: "pointer",
    "&:hover": {
      transform: "scale(1.02)",
      boxShadow: "0px 4px 15px rgba(0, 0, 0, 0.2)",
    },
  },
  moodCardContent: {
    display: "flex",
    justifyContent: "center",
  },
  moodText: {
    fontFamily: "Poppins, sans-serif",
    color: isDarkMode ? "#ffffff" : "#000000",
    marginTop: "5px",
  },
  recommendationCard: {
    backgroundColor: isDarkMode ? '#333' : '#fff',
    marginBottom: '10px',
    transition: 'transform 0.2s ease',
    '&:hover': {
      transform: 'scale(1.02)'
    }
  },
  cardContentContainer: {
    display: "flex",
    alignItems: "center",
    padding: "10px",
    width: "100%", // Take full width
  },
  imageContainer: {
    padding: "0 10px 0 0",
    flexShrink: 0,
  },
  albumImage: {
    width: "100px",
    borderRadius: "5px",
  },
  cardDetails: {
    flexGrow: 1,
    display: "flex",
    flexDirection: "column",
    justifyContent: "space-between",
    width: "100%",
  },
  songTitle: {
    fontWeight: 'bold',
    color: isDarkMode ? '#fff' : '#000'
  },
  artistName: {
    color: isDarkMode ? '#ddd' : '#666',
    marginTop: '4px'
  },
  audioPlayer: {
    width: "100%",
    marginBottom: "10px",
  },
  spotifyButton: {
    fontFamily: "Poppins, sans-serif",
  },
  timestamp: {
    color: isDarkMode ? '#aaa' : '#666',
    fontSize: '12px',
    marginTop: '4px'
  },
  emotion: {
    color: isDarkMode ? '#aaa' : '#888',
    display: 'block',
    marginTop: '8px'
  },
  moreInfo: {
    color: isDarkMode ? '#aaa' : '#666',
    display: 'block',
    textAlign: 'center',
    marginTop: '10px',
    fontStyle: 'italic'
  },
});

export default ProfilePage;
