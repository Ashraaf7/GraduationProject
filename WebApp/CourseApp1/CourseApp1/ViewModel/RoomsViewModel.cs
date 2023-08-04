using CourseApp1.data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace CourseApp1.ViewModel
{
    public class RoomsViewModel
    {
        public RoomQuestion roomQuestion { get; set; }
        public List<RoomQuestion> roomQuestionsList { get; set; }
    }
}