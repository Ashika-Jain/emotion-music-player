import { List, ListItem, ListItemText, IconButton, Paper, Typography } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';

const MoodHistory = ({ moods, onDeleteMood }) => {
  const getMoodText = (mood) => {
    if (typeof mood === 'string') {
      return mood;
    }
    if (typeof mood === 'object' && mood !== null) {
      // Handle nested emotion object
      if (mood.emotion && typeof mood.emotion === 'object') {
        return mood.emotion.label || mood.emotion.emotion || 'Unknown';
      }
      return mood.emotion || 'Unknown';
    }
    return 'Unknown';
  };

  const getTimestamp = (mood) => {
    if (typeof mood === 'object' && mood !== null && mood.timestamp) {
      return new Date(mood.timestamp).toLocaleString();
    }
    return null;
  };

  return (
    <Paper elevation={3} sx={{ padding: 2, marginBottom: 2 }}>
      <Typography variant="h6" sx={{ marginBottom: 2, fontFamily: 'Poppins' }}>
        Recent Mood History
      </Typography>
      <List>
        {moods.map((mood, index) => {
          const moodText = getMoodText(mood);
          const timestamp = getTimestamp(mood);

          return (
            <ListItem
              key={index}
              sx={{
                borderBottom: index < moods.length - 1 ? '1px solid #eee' : 'none',
                '&:hover': {
                  backgroundColor: 'rgba(0, 0, 0, 0.04)',
                },
              }}
              secondaryAction={
                <IconButton 
                  edge="end" 
                  aria-label="delete"
                  onClick={() => onDeleteMood(index)}
                  sx={{
                    color: '#ff4d4d',
                    '&:hover': {
                      backgroundColor: 'rgba(255, 77, 77, 0.1)',
                    },
                  }}
                >
                  <DeleteIcon />
                </IconButton>
              }
            >
              <ListItemText
                primary={
                  <Typography sx={{ fontFamily: 'Poppins' }}>
                    {moodText}
                  </Typography>
                }
                secondary={
                  timestamp ? (
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        color: 'text.secondary',
                        fontFamily: 'Poppins',
                        fontSize: '0.8rem'
                      }}
                    >
                      {timestamp}
                    </Typography>
                  ) : null
                }
              />
            </ListItem>
          );
        })}
      </List>
    </Paper>
  );
};

export default MoodHistory;
