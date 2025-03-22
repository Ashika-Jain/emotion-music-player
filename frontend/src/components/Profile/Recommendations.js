import {
  Box,
  Button,
  Card,
  CardContent,
  Paper,
  Typography,
  IconButton,
} from "@mui/material";
import DeleteIcon from '@mui/icons-material/Delete';
import { useState, useEffect } from 'react';

// Default album art as data URL - a simple music note icon
const defaultAlbumArt = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2NjY2NjYyIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik05IDEydjdhNCA0IDAgMSAwIDQtNFY0bDYgNSIvPjwvc3ZnPg==";

const Recommendations = ({ recommendations, onDeleteRecommendation }) => {
  // Function to get album art URL
  const getAlbumArtUrl = (rec) => {
    // Try to get the image URL from various possible locations
    if (rec.album_art) return rec.album_art;
    if (rec.image_url) return rec.image_url;
    if (rec.album?.images?.[0]?.url) return rec.album.images[0].url;
    if (rec.track?.album?.images?.[0]?.url) return rec.track.album.images[0].url;
    
    // Return a default music note SVG data URL if no image is available
    return `data:image/svg+xml;base64,${btoa(`
      <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9 18V5l12-2v13"/>
        <circle cx="6" cy="18" r="3"/>
        <circle cx="18" cy="16" r="3"/>
      </svg>
    `)}`;
  };

  // State to track image URLs
  const [imageUrls, setImageUrls] = useState({});

  // Fetch album art for all recommendations on component mount
  useEffect(() => {
    const fetchAllAlbumArt = async () => {
      const urls = {};
      for (const rec of recommendations) {
        urls[rec.track_id] = await getAlbumArtUrl(rec);
      }
      setImageUrls(urls);
    };
    fetchAllAlbumArt();
  }, [recommendations]);

  return (
    <Paper elevation={4} style={styles.resultsContainer}>
      <Typography
        variant="h6"
        style={{ fontFamily: "Poppins", marginBottom: "10px" }}
      >
        Recent Recommendations
      </Typography>
      <Box sx={styles.recommendationsList}>
        {recommendations.length > 0 ? (
          recommendations.map((rec, index) => (
            <Card key={index} sx={styles.recommendationCard}>
              <Box sx={styles.cardContentContainer}>
                {/* Left Half: Image */}
                <Box sx={styles.imageContainer}>
                  <img
                    src={getAlbumArtUrl(rec)}
                    alt={rec.track_name || rec.name || "Album Art"}
                    style={styles.albumImage}
                    onError={(e) => {
                      e.target.onerror = null; // Prevent infinite loop
                      e.target.src = getAlbumArtUrl({}); // Use default SVG
                    }}
                  />
                </Box>

                {/* Right Half: Song Details */}
                <CardContent sx={styles.cardDetails}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Box>
                      <Typography variant="subtitle1" style={styles.songTitle}>
                        {rec.track_name || rec.name}
                      </Typography>
                      <Typography variant="body2" style={styles.artistName}>
                        {rec.artist_name || rec.artist || rec.artists?.map(a => a.name).join(", ")}
                      </Typography>
                      {rec.emotion && (
                        <Typography variant="caption" style={styles.emotion}>
                          Mood: {rec.emotion}
                        </Typography>
                      )}
                    </Box>
                    <IconButton
                      onClick={() => onDeleteRecommendation(index)}
                      sx={{
                        color: '#ff4d4d',
                        '&:hover': {
                          backgroundColor: 'rgba(255, 77, 77, 0.1)',
                        },
                      }}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                  {rec.preview_url && (
                    <audio controls style={styles.audioPlayer}>
                      <source src={rec.preview_url} type="audio/mpeg" />
                      Your browser does not support the audio element.
                    </audio>
                  )}
                  {rec.external_url && (
                    <Button
                      href={rec.external_url}
                      target="_blank"
                      variant="contained"
                      style={styles.spotifyButton}
                    >
                      Listen on Spotify
                    </Button>
                  )}
                </CardContent>
              </Box>
            </Card>
          ))
        ) : (
          <Typography
            variant="body2"
            style={{
              color: "#999",
              marginTop: "20px",
              textAlign: "center",
              font: "inherit",
              fontSize: "14px",
            }}
          >
            No recommendations available. Try getting new recommendations by selecting a mood.
          </Typography>
        )}
      </Box>
    </Paper>
  );
};

const styles = {
  resultsContainer: {
    padding: "20px",
    borderRadius: "12px",
    width: "100%",
    maxWidth: "1000px",
    boxShadow: "0px 4px 20px rgba(0, 0, 0, 0.1)",
  },
  recommendationsList: {
    display: "flex",
    flexDirection: "column",
    gap: "15px",
    padding: "10px 0",
  },
  recommendationCard: {
    width: "100%",
    borderRadius: "10px",
    padding: "15px",
    boxShadow: "0px 2px 10px rgba(0, 0, 0, 0.15)",
    transition: "transform 0.2s ease",
    "&:hover": {
      transform: "translateY(-2px)",
    },
  },
  cardContentContainer: {
    display: "flex",
    flexDirection: "row",
    width: "100%",
  },
  imageContainer: {
    flex: 1,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    maxWidth: "150px",
  },
  albumImage: {
    width: "100%",
    height: "auto",
    maxHeight: "150px",
    objectFit: "cover",
    borderRadius: "10px",
    boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.2)",
  },
  cardDetails: {
    flex: 2,
    display: "flex",
    flexDirection: "column",
    paddingLeft: "20px",
  },
  songTitle: {
    fontFamily: "Poppins",
    fontSize: "1.1rem",
    fontWeight: "600",
    marginBottom: "5px",
  },
  artistName: {
    fontFamily: "Poppins",
    fontSize: "0.9rem",
    color: "#666",
    marginBottom: "10px",
  },
  emotion: {
    fontFamily: "Poppins",
    fontSize: "0.8rem",
    color: "#888",
    marginBottom: "10px",
  },
  audioPlayer: {
    width: "100%",
    marginTop: "10px",
    marginBottom: "10px",
  },
  spotifyButton: {
    backgroundColor: "#1DB954",
    color: "#fff",
    fontFamily: "Poppins",
    textTransform: "none",
    "&:hover": {
      backgroundColor: "#1ed760",
    },
  },
};

export default Recommendations;
